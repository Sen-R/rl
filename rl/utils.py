
def soft_update(local_network, target_network, tau):
    """
    Soft update parameters of target network towards local network,
    using soft update parameter `tau`.
    """
    for l_param, t_param in zip(local_network.parameters(),
                                target_network.parameters()):
        t_param.detach().copy_(tau * l_param + (1. - tau) * t_param)

def hard_update(local_network, target_network):
    """
    Replace parameters of target network by parameters of local network.
    """
    target_network.load_state_dict(local_network.state_dict())
        
