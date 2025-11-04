import torch
from PIL import Image
from src.autoencoder import StandardAutoencoder


def image_to_embedding_hf(pipeline, img):
    """
    Convert an image to its embedding using a Hugging Face pipeline.
    """
    image = Image.fromarray(img.astype('uint8'))
    features = pipeline(image)
    embedding = torch.tensor(features[0][0])
    return embedding


def generate_embeddings(model, x, device):
    """
    Generates the latent embeddings (z_i) for all frames in the dataset.

    :param model: The trained StandardAutoencoder instance.
    :param dataloader: The DataLoader using ZarrDataset (yields single frames).
    :param device: The device (e.g., 'cpu' or 'cuda') to run the model on.
    :returns: A torch.Tensor containing all generated embeddings.
    """
    model.eval() 
    
    with torch.no_grad():
        x = x.to(device)
        if isinstance(model, StandardAutoencoder): z = model(x)[0]
        else: z = model(x)
    return z.cpu()


def embs_from_dataloader(model, dataloader, device):
    """
    Generates the latent embeddings (z_i) for all frames in the dataset.

    :param model: The trained StandardAutoencoder instance.
    :param dataloader: The DataLoader using ZarrDataset (yields single frames).
    :param device: The device (e.g., 'cpu' or 'cuda') to run the model on.
    :returns: A torch.Tensor containing all generated embeddings.
    """
    model.eval() 
    all_embeddings = None
    for batch in dataloader:
        z = generate_embeddings(model, batch, device)
        all_embeddings = torch.cat((all_embeddings, z), dim=0) if all_embeddings is not None else z
    return all_embeddings