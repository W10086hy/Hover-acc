"""Simulator-only origin reward; never called by deployed control."""
import torch


def bounded(square):
    return square/(1+square)


def origin_reward(error, tilt, rate, speed, normalized_action, motor_change, failure, cfg, dt):
    """Position proximity dominates. All living rates remain positive.

    This avoids rewarding premature termination to escape a large negative
    position cost. Failure incurs an immediate additional penalty.
    """
    position_cost = error[:, :2].square().sum(-1)/cfg["xy_scale_m"]**2 + error[:, 2].square()/cfg["height_scale_m"]**2
    secondary = (cfg["tilt_bounded"]*bounded((tilt/cfg["tilt_scale_rad"]).square())+
                 cfg["rate_bounded"]*bounded((rate/cfg["rate_scale_radps"]).square())+
                 cfg["speed_bounded"]*bounded((speed/cfg["speed_scale_mps"]).square()))
    action_cost = (cfg["normalized_residual_squared"]*normalized_action.square().mean(-1)+
                   cfg["command_change_squared"]*bounded((motor_change/.03).square().mean(-1)))
    if cfg.get("kind") == "height_priority":
        # Horizontal drift cannot attenuate the reward for holding height.
        proximity = (cfg["height_gain"]/(1+error[:, 2].square()/cfg["height_scale_m"]**2)+
                     cfg["xy_gain"]/(1+error[:, :2].square().sum(-1)/cfg["xy_scale_m"]**2))
        if cfg.get("xy_tail_gain",0.) > 0:
            # Preserve the independent8:2 components, but avoid a vanishing
            # horizontal incentive at large drift. The constant keeps this
            # extra term nonnegative inside the declared living radius.
            xy_q=error[:, :2].square().sum(-1)/cfg["xy_scale_m"]**2
            limit_q=(cfg["xy_tail_radius_m"]/cfg["xy_scale_m"])**2
            proximity=proximity+cfg["xy_gain"]*cfg["xy_tail_gain"]*(limit_q-xy_q)
    else:
        proximity = cfg["position_gain"]/(1+position_cost)
    live = cfg["alive"]+proximity-secondary-action_cost
    reward = dt*live-cfg["failure"]*failure
    return torch.nan_to_num(reward,nan=-cfg["failure"],neginf=-cfg["failure"],posinf=-cfg["failure"]), position_cost+secondary
