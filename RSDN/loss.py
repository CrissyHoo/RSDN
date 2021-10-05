import torch
from torch.autograd import Variable
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()

        self.eps = torch.Tensor([1e-6]).float().cuda(non_blocking=True)


    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    #def calculate_loss(self,opt,SR,SR_D,SR_S,GT,GT_S,GT_D):
        #他们几个的shape是btchw，所以为需要先进行一些处理，这n个frame是一起计算的，算loss的时候也要注意
        #取batchsize(n videos)*N_frames的均值
     #   loss_sum=0
      #  for batch_num in range(GT.shape[0]):#B
       #     for frame in range(GT.shape[1]):#T
        #        loss_sum+=(opt.loss_alpha*self.forward(SR_S[batch_num][frame],GT_S[batch_num][frame])+opt.loss_beta*self.forward(SR_D[batch_num][frame],GT_D[batch_num][frame])+opt.loss_gamma*self.forward(SR[batch_num][frame],GT[batch_num][frame]))
                #这里index超了，我们printbatchnum和frame看一看,哪一个是4啊啊啊啊啊
                #print("batchnum",batch_num)
                #print("frame",frame)肯定是frame的原因，有一个video他没有这个frame
        #    loss_sum/=GT.shape[1]

        #losssum就这样求完了，但是需要return一个均值,上一行在某一个数据上index超了？
        #return loss_sum/(opt.trainbatchsize*GT.shape[1])
        #return loss_sum


