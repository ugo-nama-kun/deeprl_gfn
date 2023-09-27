def get_args(is_rule):
    extend_args = dict()
    extend_args["blue_nutrient"] = (0.1, 0)
    extend_args["red_nutrient"] = (0, 0.1)
    reward_weight = (1.0, 1.0)
    extend_args["reward_weight"] = reward_weight
    
    if is_rule == "CD":
        extend_args["step_IS_rule"] = "default"
        extend_args["n_blue"] = 10
        extend_args["n_red"] = 10
        
    elif is_rule == "NI": 
        extend_args["step_IS_rule"] = "default-NI-red"
        extend_args["n_blue"] = 10
        extend_args["n_red"] = 10
    
    elif is_rule == "ED":
        extend_args["step_IS_rule"] = "exchange"
        extend_args["n_blue"] = 10
        extend_args["n_red"] = 10
        
    else:
        print("@@@ {} is not defined for rule".format(is_rule))
        exit(1)
    
    return extend_args
