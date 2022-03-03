import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pdb import set_trace as bp
from operations import *


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args, distill=False):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args
        self._distill = distill
        self._kl = nn.KLDivLoss().cuda()
        self.log_latency = torch.zeros((1,), requires_grad=True, device='cuda')
        self.log_flops = torch.zeros((1,), requires_grad=True, device='cuda')
        self.log_segm = torch.zeros((1,), requires_grad=True, device='cuda')
        
        self.optimizers = [
            torch.optim.Adam((arch_param + [self.log_latency] + [self.log_flops] + [self.log_segm]),  lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
             for arch_param in self.model._arch_parameters ]
        self.latency_weight = args.latency_weight
        self.flops_weight = args.flops_weight
        self.latency_supernet = 0
        self.latency_weight = args.latency_weight
        self.flops_weight = args.flops_weight

        self.Latency_target = args.Latency_target
        self.Flops_target = args.Flops_target
        self.Latency_precision = args.Latency_precision
        self.Flops_precision = args.Flops_precision
        self.Segm_precision = args.Segm_precision
        assert len(self.latency_weight) == len(self.optimizers)

        self.latency = 0

        print("architect initialized!")

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta=None, network_optimizer=None, unrolled=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if unrolled:
                loss = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
                loss, loss_latency, loss_flops = self._backward_step(input_valid, target_valid)


        # 构造回归损失函数
        self.Latency_precision = torch.exp(-self.log_latency).cuda()
        self.Flops_precision = torch.exp(-self.log_flops).cuda()
        self.Segm_precision = torch.exp(-self.log_segm).cuda()
        diff_latency = (loss_latency - self.Latency_target) ** 2
        diff_latency = diff_latency.cuda()
        diff_Flops = (loss_flops - self.Flops_target) ** 2
        diff_Flops = diff_Flops.cuda()
        diff_segm = loss.cuda()
        L = torch.sum(self.Latency_precision * diff_latency + self.log_latency, -1) + torch.sum(self.Flops_precision * diff_Flops + self.log_flops, -1) + torch.sum(self.Segm_precision * diff_segm + self.log_segm, -1)

        loss.backward()
        if  (loss_latency - self.Latency_target) != 0: loss_latency.backward()
        if (loss_flops - self.Flops_target) != 0: loss_flops.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return torch.mean(L)

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss_latency = 0
        loss_flops = 0
        self.latency_supernet = 0
        self.flops_supernet = 0
        self.model.prun_mode = None
        for idx in range(len(self.optimizers)):
            self.model.arch_idx = idx
            if self.latency_weight[idx] == 0:
                latency = 0
                if len(self.model._width_mult_list) == 1:
                    r0 = 1./500; r1 = 499./500
                    latency = latency + r0 * self.model.forward_latency((3, 1024, 2048), alpha=True, beta=False, ratio=False)
                    latency = latency + r1 * self.model.forward_latency((3, 1024, 2048), alpha=False, beta=True, ratio=False)
                else:
                    r0 = 1./500; r1 = 497./500; r2 = 2./500
                    latency = latency + r0 * self.model.forward_latency((3, 1024, 2048), alpha=True, beta=False, ratio=False)
                    latency = latency + r1 * self.model.forward_latency((3, 1024, 2048), alpha=False, beta=True, ratio=False)
                    latency = latency + r2 * self.model.forward_latency((3, 1024, 2048), alpha=False, beta=False, ratio=True)
                self.latency_supernet = latency
                loss_latency = loss_latency + latency * self.latency_weight[idx]
        for idx_flops in range(len(self.optimizers)):
            self.model.arch_idx = idx_flops
            if self.flops_weight[idx_flops] == 0:
                flops = 0
                if len(self.model._width_mult_list) == 1:
                    # r0 = 1/2; r1 = 1/2
                    r0 = self._args.alpha_weight
                    r1 = self._args.beta_weight
                    flops = flops + r0 * self.model.forward_flops((3, 1024, 2048), alpha=True, beta=False, ratio=False)
                    flops = flops + r1 * self.model.forward_flops((3, 1024, 2048), alpha=False, beta=True, ratio=False)
                else:
                    # r0 = 1/3; r1 = 1/3; r2 = 1/3
                    r0 = self._args.alpha_weight
                    r1 = self._args.beta_weight
                    r2 = self._args.ratio_weight
                    flops = flops + r0 * self.model.forward_flops((3, 1024, 2048), alpha=True, beta=False, ratio=False)
                    flops = flops + r1 * self.model.forward_flops((3, 1024, 2048), alpha=False, beta=True, ratio=False)
                    flops = flops + r2 * self.model.forward_flops((3, 1024, 2048), alpha=False, beta=False, ratio=True)

                self.flops_supernet = flops
                if flops < 1.3e11:
                   self._args.Flops = 0
                loss_flops = loss_flops + flops * self.flops_weight[idx_flops]


        return loss, loss_latency, loss_flops

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        return unrolled_loss

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

