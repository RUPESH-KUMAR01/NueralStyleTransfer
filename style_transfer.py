import argparse
import torch
import torch.optim as optim
from model import load_image, VGGFeatureExtractor, GramMatrix, GramMSELoss, post_transform

def style_transfer(content_path, style_path, output_path, img_size=512, max_iter=100, show_iter=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Images
    content_image = load_image(content_path, img_size, device)
    style_image = load_image(style_path, img_size, device)
    input_image = torch.autograd.Variable(content_image.clone(), requires_grad=True)

    # Extract Features
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    target_layers = style_layers + content_layers

    vgg_extractor = VGGFeatureExtractor(target_layers).to(device)
    
    style_features = vgg_extractor(style_image)
    content_features = vgg_extractor(content_image)

    # Compute style targets (Gram Matrices) and content targets
    style_targets = [GramMatrix()(style_features[layer]).detach() for layer in style_layers]
    content_targets = [content_features[layer].detach() for layer in content_layers]
    targets = style_targets + content_targets

    # Define Loss Functions and Weights
    loss_fns = [GramMSELoss()] * len(style_layers) + [torch.nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    style_weights = [1e3 / (n ** 2) for n in [64, 128, 256, 512, 512]]
    content_weights = [1.0]
    weights = style_weights + content_weights

    # Optimize the Image
    optimizer = optim.LBFGS([input_image.requires_grad_()])

    n_iter = [0]
    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            input_features = vgg_extractor(input_image)
            style_losses = [loss_fns[i](input_features[layer], targets[i]) * weights[i] for i, layer in enumerate(style_layers)]
            content_losses = [loss_fns[len(style_layers) + i](input_features[layer], targets[len(style_layers) + i]) * weights[len(style_layers) + i] for i, layer in enumerate(content_layers)]
            loss = sum(style_losses) + sum(content_losses)
            loss.backward()
            n_iter[0] += 1

            if n_iter[0] % show_iter == 0:
                print(f"Iteration {n_iter[0]}: Loss {loss.item():.4f}")
            
            return loss

        optimizer.step(closure)

    # Save Output Image
    output_image = post_transform(input_image.squeeze().cpu())
    output_image.save(output_path)
    print(f"Stylized image saved at: {output_path}")
    output_image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer using VGG19")
    parser.add_argument("content", type=str, help="Path to content image")
    parser.add_argument("style", type=str, help="Path to style image")
    parser.add_argument("output", type=str, help="Path to save the stylized output")
    parser.add_argument("--img_size", type=int, default=512, help="Resize image to this size (default: 512)")
    parser.add_argument("--max_iter", type=int, default=100, help="Number of optimization iterations (default: 100)")
    parser.add_argument("--show_iter", type=int, default=5, help="Print loss every N iterations (default: 5)")
    
    args = parser.parse_args()
    
    style_transfer(args.content, args.style, args.output, args.img_size, args.max_iter, args.show_iter)
