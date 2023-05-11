import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):

    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:

        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask


    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta



    labels_distribute = labels / labels.sum(dim=1)


    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss



def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):

    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)


    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):

    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):


    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())


    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def get_alpha_t(sims):
    evidences = torch.exp(torch.tanh(sims)/0.1)



    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    return alpha_i2t, alpha_t2i, norm_e
    
def get_alpha(sims):

    evidences = F.relu(sims)


    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    return alpha_i2t, alpha_t2i, norm_e

def compute_unc(image_fetures, text_fetures, pid, logit_scale, image_id=None):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:

        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask



    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = t2i_cosine_theta



    labels_distribute = labels / labels.sum(dim=1)
    alpha_t2i, alpha_i2t, _ = get_alpha(text_proj_image)
    alpha_t2i_t, alpha_i2t_t, _ = get_alpha_t(text_proj_image)
    u_1 = batch_size / torch.sum((1.0 * alpha_i2t_t + 0.0 * alpha_i2t) / 1.0, dim=1, keepdim=True)
    








        



    loss = torch.mean(mse_loss_tanh(labels_distribute, alpha_t2i_t, batch_size, 0.1)) + 1.0 * torch.mean(mse_loss_tanh(labels_distribute, alpha_i2t_t, batch_size, 0.1))
    loss1 = torch.mean(mse_loss(labels, alpha_t2i, batch_size, 0.1)) + 1.0 * torch.mean(mse_loss(labels, alpha_i2t, batch_size, 0.1))
    loss = 0.5 * loss + 0.5 * loss1

    return loss

def kl_divergence(alpha):
    beta = torch.ones([1, self.num_classes], dtype=torch.float32).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

def KL(alpha, c):
    beta = torch.ones((1, c)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    E = alpha - 1
    m = alpha / S

    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)



    return (A + B) + C
    
def mse_loss_tanh(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    E = alpha - 1
    m = alpha / S

    u_1 = alpha.size(0) / S
    min_u = torch.min(u_1)
    max_u = torch.max(u_1)
    u_norm = (u_1 - min_u) / (max_u - min_u)


    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)



    return (A + B) + C


def compute_id_unc(image_fetures, text_fetures, pid, logit_scale, label, image_id=None):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:

        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask



    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = t2i_cosine_theta



    labels_distribute = labels / labels.sum(dim=1)
    alpha_t2i, alpha_i2t, _ = get_alpha(text_proj_image)




    loss = torch.mean(ce_loss(label, alpha_t2i, batch_size, 0.1)) + 1.0 * torch.mean(ce_loss(label, alpha_i2t, batch_size, 0.1))

    return loss

def ce_loss(label, alpha, c, lambda2):



        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_score = alpha / S
        loss_cls = F.nll_loss(torch.log(pred_score), label, reduction='none')
        


        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        


        kl_alpha = (alpha - 1) * (1 - label) + 1
        kl_div = lambda2 * \
                 KL(kl_alpha, c)
        losses = loss_cls + loglikelihood_var + kl_div
        return losses


def cosine_sim(im, s):
    """Cosine similarity between all the two pairs
    """
    return im.mm(s.t())


def l1_sim(im, s):
    """l1 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=1)
    return scro


def l2_sim(im, s):
    """L2 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro


def msd_loss(im, s):
    """MSD similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro.pow(2)

def calculate_similarity(image_embedding, text_embedding):
    image_embedding = image_embedding.view(image_embedding.size(0), -1)
    text_embedding = text_embedding.view(text_embedding.size(0), -1)
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    similarity_match = torch.sum(image_embedding_norm * text_embedding_norm, dim=1)

    return similarity.float(), similarity_match

class IntraLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, up=0.2, down=0.04, lamb=1.0):
        super(IntraLoss, self).__init__()

        self.margin = margin
        self.measure = measure
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'msd':
            self.sim = msd_loss
        elif self.measure == 'l1':
            self.sim = l1_sim
        elif self.measure == 'l2':
            self.sim = l2_sim
        self.max_violation = max_violation
        self.up = up
        self.down = down
        self.lamb = lamb

    def forward(self, img_emb, text_emb):

        mx, mx1 = calculate_similarity(img_emb, text_emb)
        scores = self.sim(mx, mx)



        if self.measure == 'cosine':
            diagonal = scores.diag()

            scores = scores

            eye = torch.eye(scores.size(0)).float()
            scores_non_self = scores - eye


            scores_non_self = scores_non_self * (
                scores_non_self.gt(self.up).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(1 - self.down).float())
            scores_norm = scores_non_self.sum() / scores.size(0)



        elif self.measure == 'msd' or self.measure == 'l1' or self.measure == 'l2':
            scores_non_self = torch.nn.functional.normalize(scores)



            idx_up = round(self.up * scores.size(0))
            idx_down = round(self.down * scores.size(0))
            _, s_index = scores_non_self.sort()
            s_mean = scores_non_self.mean()

            s_up = scores_non_self[0, s_index[0, idx_up]]
            s_down = scores_non_self[0, s_index[0, idx_down]]


            scores_non_self = scores_non_self * (
                scores_non_self.gt(s_down).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(s_up).float())
            scores_norm = scores_non_self.sum() / scores.size(0)



        return self.lamb * scores_norm

def compute_intra(image_embeddings, text_embeddings):
    intra_loss_f = IntraLoss(measure='l1', up=0.22, down=0.05)
    intra_loss_f_t = IntraLoss(measure='l1', up=0.52, down=0.05)
    intra_lossg = intra_loss_f(image_embeddings, image_embeddings)
    intra_lossg_t = intra_loss_f(text_embeddings, text_embeddings)
    return intra_lossg + intra_lossg_t
    
def compute_intra_g(image_embeddings, text_embeddings):
    intra_loss_f = IntraLoss(measure='l1', up=0.13, down=0.05)
    intra_loss_f_t = IntraLoss(measure='l1', up=0.16, down=0.05)
    intra_lossg = intra_loss_f(image_embeddings, image_embeddings)
    intra_lossg_t = intra_loss_f(text_embeddings, text_embeddings)
    return intra_lossg + intra_lossg_t
    

