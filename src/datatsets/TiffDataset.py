from torch.utils.data import Dataset, DataLoader

class NpzDataset(Dataset):
    def __init__(self, data_list_path):
        with open(data_list_path, "r") as f:
            data_lsit = json.load(f)
        self.data_list = data_lsit
        
        self.length = len(self.data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        norm_metadata = data["norm_metadata"]
        prompt_embeds = data["prompt_embeds"]
        latent_image = data["latent_image"]
        cond = data["cond"]
        return {
            "norm_metadata": norm_metadata,
            "prompt_embeds": prompt_embeds,
            "latent_image": latent_image,
            "cond": cond  
        }

def custom_collate_fn(batch):
    norm_metadata_list = [torch.from_numpy(item["norm_metadata"]) for item in batch]
    prompt_embeds_list = [torch.from_numpy(item["prompt_embeds"]) for item in batch]
    latent_image_list = [torch.from_numpy(item["latent_image"]) for item in batch]
    cond_list = [torch.from_numpy(item["cond"]) for item in batch]
    
    norm_metadata = torch.stack(norm_metadata_list)
    latent_image = torch.stack(latent_image_list)
    cond_l = torch.stack(cond_list)    
    prompt_embeds = prompt_embeds_list
    cond = [cond_l]
    
    return {
        "norm_metadata": norm_metadata,
        "prompt_embeds": prompt_embeds,
        "latent_image": latent_image,
        "cond": cond
    }