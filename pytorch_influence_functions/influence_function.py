#! /usr/bin/env python3

import torch
from torch.autograd import grad
from pytorch_influence_functions.utils import display_progress, concate_list_to_array
import numpy as np
import torch.nn as nn
from scipy.optimize import fmin_ncg


def s_test(v, model, params, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000, print_iter=100):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        #########################
        # Done: do x, t really have to be chosen from the train sest
        #########################
        x, t = z_loader.dataset[i]
        x = z_loader.collate_fn([x])
        t = z_loader.collate_fn([t])

        model.eval()
        if gpu >= 0:
            x, t = x.cuda(), t.cuda()
        y = model(x)
        loss = calc_loss(y, t)
        hv = hvp(loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate = [
            _v + (1 - damp) * _h_e - _hv / scale
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        # display_progress("Calc. s_test recursions: ", i, recursion_depth)
        # Caculate norm of params
        if (i % print_iter == 0) or (i == recursion_depth - 1):
            print("Recursion at depth %s: norm is %.8lf" % (i,
                    np.linalg.norm(concate_list_to_array(h_estimate))))
    
    return h_estimate


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, t)
    # y = torch.nn.functional.log_softmax(y)
    # loss = torch.nn.functional.nll_loss(
    #     y, t, weight=None, reduction='mean')
    return loss


def grad_z(z, t, model, params, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    # params = [ p for p in model.parameters() if p.requires_grad ][-2:] # only for cls
    return list(grad(loss, params, create_graph=False))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(outputs=y, inputs=w, create_graph=True, retain_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # elemwise_products = [
    #     grad_elem.mul(v_elem)
    #     for grad_elem, v_elem in zip(first_grads, v)
    # ]

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=False)

    # norm = 0
    # for ele_ in elemwise_products:
    #     norm += ele_.pow(2).sum()
    # norm = norm.sqrt()
    
    # return_grads = grad(norm, w)

    return return_grads


def minibatch_hessian_vector_val(v, model, z_loader, params, gpu=-1, damp=0.01):
    """
    Reference to pangwei/tf1.1
    Arguments:
        v: test loss
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        params: model params for caculating Hessian

    Returns:
        h_estimate: list of torch tensors, s_test"""
    num_iter = len(z_loader)
    hessian_vector_val = None

    v = [torch.from_numpy(j) for j in v]
    v = [torch.reshape(v[i], params[i].size()) for i in range(len(v))]
    v = [j.cuda() for j in v]
    for batch_idx, sample_batched in enumerate(z_loader):
        x = sample_batched[0]
        t = sample_batched[1]

        model.eval()
        if gpu >= 0:
            x, t = x.cuda(), t.cuda()
        y = model(x)
        loss = calc_loss(y, t)
        hessian_vector_val_tmp = hvp(loss, params, v)

        if hessian_vector_val is None:
            hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_tmp]
        else:
            hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_tmp)]

    hessian_vector_val = [a + damp * b for (a,b) in zip(hessian_vector_val, v)]
    hessian_vector_val = [j.detach().cpu().numpy().ravel() for j in hessian_vector_val]

    return hessian_vector_val


def vect_to_list(vect, params):
    start = 0
    vect_list = []
    for i in range(len(params)):
        end = len(params[i].detach().cpu().numpy().flatten()) + start
        vect_list.append(np.array(vect[start:end]))
        start = end
    
    return vect_list


def get_fmin_loss_fn(v):

    def get_fmin_loss(x, model, z_loader, params, gpu, damp):
        hessian_vector_val = minibatch_hessian_vector_val(vect_to_list(x, params), 
                                                        model, z_loader, params, gpu, damp)

        return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
    
    return get_fmin_loss


def get_fmin_grad_fn(v):

    def get_fmin_grad(x, model, z_loader, params, gpu, damp):
        hessian_vector_val = minibatch_hessian_vector_val(vect_to_list(x, params), 
                                                        model, z_loader, params, gpu, damp)

        return np.concatenate(hessian_vector_val) - np.concatenate(v)

    return get_fmin_grad


def get_fmin_hvp(x, p, model, z_loader, params, gpu, damp):
    hessian_vector_val = minibatch_hessian_vector_val(vect_to_list(p, params), 
                                                        model, z_loader, params, gpu, damp)

    return np.concatenate(hessian_vector_val)


def get_inverse_hvp_cg(v, model, z_loader, params, gpu=-1, damp=0.01):
    v = [i.detach().cpu().numpy().ravel() for i in v]
    fmin_loss_fn = get_fmin_loss_fn(v)
    fmin_grad_fn = get_fmin_grad_fn(v)
    
    fmin_results = fmin_ncg( 
        f=fmin_loss_fn,
        x0=np.concatenate(v),
        fprime=fmin_grad_fn,
        fhess_p=get_fmin_hvp,
        #callback=cg_callback,
        avextol=1e-8,
        maxiter=100,
        args=(model, z_loader, params, gpu, damp))

    fmin_results = vect_to_list(fmin_results, params)
    fmin_results = [torch.from_numpy(j) for j in fmin_results]
    fmin_results = [torch.reshape(fmin_results[i], params[i].size()) for i in range(len(fmin_results))]
    fmin_results = [j.cuda() for j in fmin_results]
    return fmin_results
