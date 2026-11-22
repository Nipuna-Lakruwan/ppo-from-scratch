import torch

from ppo import ActorCritic, gae


def test_gae_single_step_matches_td_error():
    r = torch.tensor([[1.0]])
    v = torch.tensor([[0.5]])
    d = torch.tensor([[0.0]])
    adv, ret = gae(r, v, d, last_value=torch.tensor([2.0]), last_done=torch.tensor([0.0]), gamma=0.9, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[1.0 + 0.9 * 2.0 - 0.5]]))
    assert torch.allclose(ret, adv + v)


def test_gae_stops_at_episode_end():
    r = torch.tensor([[0.0], [1.0]])
    v = torch.zeros(2, 1)
    d = torch.tensor([[0.0], [1.0]])  # a new episode starts at t=1
    adv, _ = gae(r, v, d, last_value=torch.tensor([0.0]), last_done=torch.tensor([0.0]), gamma=1.0, lam=1.0)
    assert adv[0, 0] == 0.0  # the reward from the next episode must not leak back


def test_actor_critic_shapes():
    ac = ActorCritic(8, 4)
    a, logp, ent, v = ac.act(torch.randn(5, 8))
    assert a.shape == logp.shape == ent.shape == v.shape == (5,)
