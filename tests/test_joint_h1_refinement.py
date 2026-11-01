"""验证最终图像梯度到达结构目标，并与原推理一步更新一致。"""
import torch
from models.event_structure_guidance import GuidanceConfig, StructureObservation, freeze_observation_masks, optimize_features
from models.image_structure import StructureOperatorConfig,image_structure
from models.joint_h1_refinement import differentiable_guidance
from models.joint_h1_refinement import joint_observation
from models.event_structure_head import EventStructurePredictor,StructureHeadConfig
import copy


def test_joint_gradient_and_inference_parity():
    torch.manual_seed(2)
    feature=torch.randn(1,3,16,16)*.15
    # Huber 线性区对目标的混合二阶导为零；用二次区检查可学习目标路径。
    target=torch.nn.Parameter(image_structure((torch.tanh(feature*.8)+1)/2).detach()+.01)
    fusion=torch.nn.Parameter(torch.tensor(.8))
    observation=freeze_observation_masks(StructureObservation(target,torch.ones_like(target)*.7))
    cfg=GuidanceConfig(steps=1)
    def render(correction):
        x=feature if correction is None else correction.apply(feature)
        return torch.tanh(x*fusion)
    out,stats=differentiable_guidance(render,observation,cfg,StructureOperatorConfig())
    assert stats['accepted_steps']==1
    objective=out.square().mean()
    gradients=torch.autograd.grad(objective,(target,fusion))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    reference,_=optimize_features(render,observation,cfg,StructureOperatorConfig())
    torch.testing.assert_close(out.detach(),reference,rtol=1e-5,atol=1e-7)


def test_joint_rejection_falls_back():
    torch.manual_seed(3)
    feature=torch.randn(1,3,16,16)*.15
    obs=freeze_observation_masks(StructureObservation(torch.full((1,2,16,16),.4),torch.full((1,2,16,16),.7)))
    cfg=GuidanceConfig(steps=1,max_low_change=0,max_outside_change=0)
    def render(c):return feature if c is None else c.apply(feature)
    out,stats=differentiable_guidance(render,obs,cfg,StructureOperatorConfig())
    assert stats['accepted_steps']==0
    torch.testing.assert_close(out,feature)


def test_fixed_teacher_masks_despite_trainable_shared_features():
    torch.manual_seed(4)
    head=EventStructurePredictor(StructureHeadConfig(base_channels=4))
    teacher=copy.deepcopy(head).requires_grad_(False)
    events=torch.rand(1,10,32,32)
    a,_=joint_observation(head,teacher,events,.25)
    with torch.no_grad():head.enc1[0].weight.add_(.2)
    b,_=joint_observation(head,teacher,events,.25)
    assert not torch.equal(a.edges,b.edges)
    for name in ['confidence','feature_mask','protection_mask']:
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
