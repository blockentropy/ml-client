import argparse
import re
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

def rename_keys(input_file):
    # Load the safetensors file
    state_dict = safe_load_file(input_file)

    # Function to rename a key
    def rename_key(key):
        # Remove 'base_model.model.' prefix
        new_key = re.sub(r'^base_model\.model\.', '', key)
        return new_key

    def keep_layer(layer_num):
        if 20 <= layer_num <= 39 or 60 <= layer_num <= 79 or 100 <= layer_num:
            return True
        return False

    # Create a new dictionary with renamed keys
    renamed_dict = {}
    for k, v in state_dict.items():
        new_key = rename_key(k)
        i = new_key.find("model.layers.") 
        if i != -1:
            match = re.search(r"model\.layers\.(\d+)", new_key)
            if match:
                layer_num = int(match.group(1))
                if keep_layer(layer_num):
                    renamed_dict[new_key] = v

    return renamed_dict

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Rename keys in a safetensors file.")
    parser.add_argument("input_file", help="Path to the input safetensors file")
    parser.add_argument("output_file", help="Path to save the modified safetensors file")
    
    # Parse arguments
    args = parser.parse_args()

    # Rename keys
    renamed_state_dict = rename_keys(args.input_file)

    # Print the renamed keys
    print("Renamed keys:")
    for key in renamed_state_dict.keys():
        print(key)

    # Save the renamed state_dict to the output file
    safe_save_file(renamed_state_dict, args.output_file)
    print(f"\nModified safetensors file saved to: {args.output_file}")

if __name__ == "__main__":
    main()
