import torch
from recstudio.model import basemodel, scorer
from recstudio.data.advance_dataset import ALSDataset


class GALS(basemodel.BaseRetriever):
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.I_u = train_data.user_count
        self.N = train_data.num_items - 1
        self.M = train_data.num_users - 1
        self.I_u_2N = 0.5 * self.I_u / self.N

    def _get_dataset_class():
        return ALSDataset

    def _get_train_loaders(self, train_data):
        train_config = self.config['train']
        loader = train_data.train_loader(
            batch_size=train_config['batch_size'],
            shuffle=True,
            drop_last=False)
        loader_T = train_data.transpose().train_loader(
            batch_size=train_config['batch_size'],
            shuffle=True,
            drop_last=False)
        return [loader, loader_T]  # use combine loader or concate loaders

    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[nepoch % len(self.trainloaders)], False

    def _get_optimizers(self):
        return None

    def training_epoch(self, nepoch):
        if  nepoch % 2 == 0:
            self.QtQ = self.item_encoder.weight.T @ self.item_encoder.weight
        else:
            WP = (self.Iu_2N.view(-1, 1) * self.query_encoder.weight)
            self.PtWP = self.query_encoder.weight.T @ WP
        return super().training_epoch(nepoch)

    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad = False
        self.item_encoder.weight.requires_grad = False
        self.register_buffer('eye', self.config['train']['lambda'] * torch.eye(self.embed_dim))
        self.register_buffer('Iu', self.I_u)
        self.register_buffer('Iu_2N', self.I_u_2N)

    def _get_loss_func(self):
        return None

    def _get_score_func(self):
        return scorer.InnerProductScorer()
    
    @torch.no_grad()
    def training_step(self, batch):
        if batch[self.fuid].dim() == 1:  # user model, updating user embedding
            item_embed = self.item_encoder(self._get_item_feat(batch))  # B x n x D
            Iu_2N = self.Iu_2N[batch[self.fuid]]
            bQQ = torch.bmm(item_embed.transpose(1, 2), item_embed) * self.config['train']['alpha']
            QWuQ = torch.einsum('b,ij->bij', Iu_2N, self.QtQ) + bQQ  # BxDxD
            sum_W = self.Iu[batch[self.fuid]] * 0.5 + self.config['train']['alpha'] * self.N    # B
            QWuQ += torch.einsum('b,ij->bij', sum_W, self.eye)
            QWuQ += self.eye
            RWuQ = torch.einsum('b,ij->bij', - Iu_2N, self.item_encoder.weight).sum(1)  # BxD
            RWuQ = RWuQ + ((0.5 + self.config['train']['alpha']) * item_embed).sum(1) # BxD
            output = torch.linalg.solve(QWuQ, RWuQ)
            if self.config['model']['item_bias']:
                output[:, -1] = 1.0
            self.query_encoder.weight[batch[self.fuid]] = output  # B x D
        else:
            user_embed = self.query_encoder(self._get_query_feat(batch))
            Iu_2N = self.Iu_2N[batch[self.fuid]]  # BxN
            bPP = torch.bmm(user_embed.transpose(1, 2), user_embed) * self.config['train']['alpha']
            PiP = self.PtWP + bPP  # BxDxD
            sum_W = self.Iu_2N.sum() + self.config['train']['alpha'] * self.M    # 1

            PiP = PiP + sum_W * self.eye  # BxDxD
            # PiP = PiP.view(1, *PiP.shape).repeat(user_embed.size(0), 1, 1)  # BxDxD

            RWiP = self.Iu_2N @ self.query_encoder.weight  # D

            RWiP = RWiP.view(1, -1) + (0.5 + self.config['train']['alpha']) * user_embed.sum(1)  # BxD
            output = torch.linalg.solve(PiP, RWiP)
            self.item_encoder.weight[batch[self.fiid]] = output
        loss = torch.tensor(0.0)

        return {'loss': loss}
