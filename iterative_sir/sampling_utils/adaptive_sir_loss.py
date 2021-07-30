import torch
#from scipy.stats import gamma, invgamma


def get_optimizer(parameters, optimizer = "Adam", lr = 1e-3, weight_decay=1e-5):
    if optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def get_loss(loss):
    if loss == "mix_kl":
        return mix_kl
    if loss == "forward_kl":
        return forward_kl
    if loss == "backward_kl":
        return backward_kl
    else:
        raise NotImplementedError

###Write here f divergence

#write here forward/backward KL

def entropy(target, proposal, flow, y, acc_rate=1.):
    u = proposal.sample(y.shape[:-1])
    x, log_jac = flow(u)

    entr = - proposal(u) + log_jac

    entr = acc_rate * entr
    grad_est = entr

    return entr.mean(), grad_est.mean()

#def log_likelihood

def importance_forward_kl(target, proposal, flow, y, batch_size=10, N=10):
    x = proposal.sample((batch_size, N))
    x_ = x.view(-1, y.shape[-1])
    log_weight = target(x_) - proposal(x_)

    log_weight = log_weight.view(batch_size, N)
    max_logs = torch.max(log_weight, dim = 1)[0][:, None]
    log_weight = log_weight - max_logs
    weight = torch.exp(log_weight)
    sum_weight = torch.sum(weight, dim = 1)
    weight = weight/sum_weight[:, None]        

    weight[weight != weight] = 0.
    weight[weight.sum(1) == 0.] = 1.

    x_inv, minus_log_jac = flow.inverse(x.view(-1, y.shape[-1]))
    #est = target(y) - proposal.log_prob(x_inv) - minus_log_jac
    minus_log_q = - proposal(x_inv) - minus_log_jac
    kl = (weight * minus_log_q.view(*weight.shape)).mean()
    return kl, kl


def forward_kl(target, proposal, flow, y):
    ##Here, y \sim \target
    ###PROPOSAL INITIAL DISTR HERE
    y_ = y.detach().requires_grad_()
    u, minus_log_jac = flow.inverse(y_)
    est = target(y) - proposal.log_prob(u) - minus_log_jac
    grad_est = - proposal.log_prob(u) - minus_log_jac
    return est.mean(), grad_est.mean()

def backward_kl(target, proposal, flow, y):
    u = proposal.sample(y.shape[:-1])
    x, log_jac = flow(u)
    est = proposal.log_prob(u) - log_jac - target.log_prob(x)
    grad_est = - log_jac - target.log_prob(x)
    return est.mean(), grad_est.mean()


def mix_kl(target, proposal, flow, y, acc_rate=1., alpha= .99, beta=.1): #.2):
    est_f, grad_est_f = forward_kl(target, proposal, flow, y)
    est_b, grad_est_b = backward_kl(target, proposal, flow, y)
    entr, grad_est_entr = entropy(target, proposal, flow, y, acc_rate=acc_rate)
    imp_f, grad_imp_f = importance_forward_kl(target, proposal, flow, y)


    if torch.isnan(grad_est_b).item():
        grad_est_b = 0

    return (alpha * est_f + (1. - alpha) * est_b  + alpha * grad_imp_f - beta * entr, 
            alpha * grad_est_f + (1. - alpha) * grad_est_b + alpha * grad_imp_f - beta * grad_est_entr)

