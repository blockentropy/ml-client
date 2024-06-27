from copy import copy, deepcopy
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2 import ext

def ExLlamaV2MergePassthrough(model):
    ### STACKED LAYERS
    layer_arrangement = []

    #Venus 120b 
    layer_arrangement.extend(range(0, 20))
    layer_ranges = [
        (10, 30),  # adjusted for Python's zero-indexing and end-exclusive range.
        (20, 40),
        (30, 50),
        (40, 60),
        (50, 70),
        (60, 80),
    ]

    for start, end in layer_ranges:
        layer_arrangement.extend(range(start, end))

    old_modules = model.modules
    model.modules = old_modules[:1]
    model.modules_dict = {}

    model.modules_dict[model.modules[-1].key] = model.modules[-1]
    for i, idx in enumerate(layer_arrangement):
        model.modules += [copy(old_modules[idx*2 + 1])]
        model.modules[-1].layer_idx = i # for duplicate layers to use a different cache
        model.modules[-1].key = model.modules[-1].key.split('.')[0] + "." + model.modules[-1].key.split('.')[1] + '.' + str(i)
        model.modules_dict[model.modules[-1].key] = model.modules[-1]

        updated_submodules = []
        for m in model.modules[-1].submodules:
            copied_submodule = copy(m)
            updated_submodules.append(copied_submodule)
            # Split the key into parts
            key_parts = copied_submodule.key.split('.')
            key_parts[2] = str(i)  # Convert i to string, since we're dealing with string manipulation
            updated_key = '.'.join(key_parts)
            print(f"layer {i} is {updated_key}")
            #copied_submodule.reload()
            copied_submodule.key = updated_key
            # Check if the index exists and perform the check
            if len(key_parts) > 4:  # Ensure the list is long enough
                #copied_submodule.layer_idx = i
                copied_submodule.lora_a_tensors = {}
                copied_submodule.lora_b_tensors = {}
                if key_parts[4] == 'q_proj':
                    model.modules[-1].q_proj = copied_submodule
                if key_parts[4] == 'k_proj':
                    model.modules[-1].k_proj = copied_submodule
                if key_parts[4] == 'v_proj':
                    model.modules[-1].v_proj = copied_submodule
                if key_parts[4] == 'o_proj':
                    model.modules[-1].o_proj = copied_submodule
            #model.modules[-1].load()
            model.modules_dict[updated_key] = copied_submodule
        model.modules[-1].submodules = updated_submodules
        model_self =  model.modules[-1]
        if model_self.q_proj.is_quant():
            device_tensors = model.get_device_tensors(model_self.device_idx)
            device_tensors.begin_scratch_alloc()
            model_self.temp_state = device_tensors.get_scratch_slice(model_self.temp_state_size())
            model_self.temp_dq = device_tensors.get_scratch_slice(model_self.temp_dq_size())
            if model_self.q_norm is None:
                q_norm = none_tensor
            else:
                q_norm = self.q_norm.weight

            if model_self.k_norm is None:
                k_norm = none_tensor
            else:
                k_norm = self.k_norm.weight
            model_self.q_handle = ext_c.make_q_attn(model_self.input_layernorm.weight,
                                        model_self.input_layernorm.bias if model_self.input_layernorm.bias is not None else ext.none_tensor,
                                        isinstance(model_self.input_layernorm, ExLlamaV2RMSNorm),
                                        model_self.input_layernorm.variance_epsilon,
                                        model_self.q_proj.q_handle,
                                        model_self.k_proj.q_handle,
                                        model_self.v_proj.q_handle,
                                        model_self.o_proj.q_handle,
                                        model_self.temp_state,
                                        model_self.temp_dq,
                                        model.config.max_input_len * model.config.max_batch_size,
                                        model.config.hidden_size,
                                        model.config.num_attention_heads,
                                        model.config.num_key_value_heads,
                                        model.config.head_dim,
                                        model.config.max_seq_len,
                                        model_self.has_residual,
                                        model.config.arch.rope_style.value,
                                        q_norm,
                                        k_norm)

        print(f"layer {i} is {model.modules[-1].key}")
        model.modules += [copy(old_modules[idx*2 + 2])]
        model.modules[-1].layer_idx = i
        model.modules[-1].key = model.modules[-1].key.split('.')[0] + "." +  model.modules[-1].key.split('.')[1] + '.' + str(i)
        model.modules_dict[model.modules[-1].key] = model.modules[-1]
        updated_submodules_mlp = []
        for m in model.modules[-1].submodules:
            copied_submodule = copy(m)
            updated_submodules_mlp.append(copied_submodule)
            # Split the key into parts
            key_parts = copied_submodule.key.split('.')
            key_parts[2] = str(i)  # Convert i to string, since we're dealing with string manipulation
            updated_key = '.'.join(key_parts)
            print(f"layer {i} is {updated_key}")
            #copied_submodule.reload()
            copied_submodule.key = updated_key
            if len(key_parts) > 4:  # Ensure the list is long enough
                #copied_submodule.layer_idx = i
                copied_submodule.lora_a_tensors = {}
                copied_submodule.lora_b_tensors = {}
                if key_parts[4] == 'down_proj':
                    model.modules[-1].down_proj = copied_submodule
                if key_parts[4] == 'up_proj':
                    model.modules[-1].up_proj = copied_submodule
                if key_parts[4] == 'gate_proj':
                    model.modules[-1].gate_proj = copied_submodule
            # Update the dictionary entry with the new key
            #model.modules[-1].load()
            model.modules_dict[updated_key] = copied_submodule
        model.modules[-1].submodules = updated_submodules_mlp
        model_self =  model.modules[-1]
        if model_self.up_proj.is_quant():
            device_tensors = model.get_device_tensors(model_self.device_idx)
            device_tensors.begin_scratch_alloc()
            model_self.q_handle = ext_c.make_q_mlp(model_self.post_attention_layernorm.weight,
                                        model_self.post_attention_layernorm.bias if model_self.post_attention_layernorm.bias is not None else ext.none_tensor,
                                        isinstance(model_self.post_attention_layernorm, ExLlamaV2RMSNorm),
                                        model_self.post_attention_layernorm.variance_epsilon,
                                        model_self.gate_proj.q_handle,
                                        model_self.up_proj.q_handle,
                                        model_self.down_proj.q_handle,
                                        device_tensors.get_scratch_slice(model_self.temp_state_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_a_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_b_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_dq_size()),
                                        model.config.max_input_len * model.config.max_batch_size,
                                        model.config.arch.mlp_act_func == "gelu",
                                        model_self.has_residual)

    model.modules += old_modules[-2:]
    model.head_layer_idx = len(model.modules) -1
    model.config.num_hidden_layers = len(layer_arrangement)
    model.last_kv_layer_idx = len(model.modules) -4

    model.modules_dict[old_modules[-2].key] = model.modules[-2]
    model.modules_dict[old_modules[-1].key] = model.modules[-1]
    model.cache_map = {}
    model.set_cache_map()

    print("Keys in new model.modules_dict:")
    for key in model.modules_dict.keys():
        print(key)
    print("Num of hidden layers:" +str(model.config.num_hidden_layers))
    # Load LoRA
    #lora_directory = "../exllamav2/checkpoint-100/"
    #lora_directory = "../exllamav2/unsloth/unsloth_outputs_expand/checkpoint-7000/"
    #lora_directory = "../exllamav2/unsloth/unsloth_outputs_expand8x/checkpoint-12000/"
    #lora_directory = "../exllamav2/unsloth/unsloth_outputs_yi_lima/checkpoint-8000/"
    #lora_directory = "../exllamav2/unsloth/trained_unsloth_tinyllama_lima/" 
    #lora_directory = "../exllamav2/openhermes_out_stacked_94layers/checkpoint-11000/"

    #lora = None

    return model