import json
import os
import argparse

def migrate_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Migrating {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Helper function to recursively replace keys
    def rename_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # Replace 'ours' with 'caca' in keys
                new_key = k.replace('ours', 'caca')
                new_dict[new_key] = rename_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [rename_keys(item) for item in obj]
        else:
            return obj

    new_data = rename_keys(data)

    # Backup original file
    backup_path = file_path + ".bak"
    os.rename(file_path, backup_path)
    print(f"  - Created backup: {backup_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"  - Successfully migrated to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Migrate result JSON files from "ours" naming to "caca"')
    parser.add_argument('--dir', type=str, default='results', help='Directory containing JSON files')
    parser.add_argument('--file', type=str, help='Specific JSON file to migrate')
    
    args = parser.parse_args()

    if args.file:
        migrate_data(args.file)
    else:
        # Search in directory
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found.")
            return
            
        for filename in os.listdir(args.dir):
            if filename.endswith('.json'):
                migrate_data(os.path.join(args.dir, filename))
        
        # Also check fruit of seed search in root
        if os.path.exists('seed_search_results.json'):
            migrate_data('seed_search_results.json')

if __name__ == "__main__":
    main()
