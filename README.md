### **README for `style_transfer.py`**  

# Neural Style Transfer using PyTorch  

This project applies artistic style transfer using a **pre-trained VGG19 model** in PyTorch. The model extracts content and style features and optimizes an input image to blend them together.  

---

## **Installation**  
Make sure you have Python installed, then install the required dependencies:  
```bash
pip install torch torchvision pillow
```

---

## **Usage**  
Run the script from the command line:  
```bash
python style_transfer.py <content_image> <style_image> <output_image> [--img_size 512] [--max_iter 100] [--show_iter 5]
```

### **Example**  
```bash
python style_transfer.py Images\Tuebingen_Neckarfront.jpg Images\vangogh_starry_night.jpg output.jpg --img_size 512 --max_iter 200 --show_iter 10
```

### **Arguments**  
| Argument       | Description                                      | Default |
|---------------|--------------------------------------------------|---------|
| `content`     | Path to the content image                        | Required |
| `style`       | Path to the style image                          | Required |
| `output`      | Path to save the stylized output                 | Required |
| `--img_size`  | Resize image for processing                      | 512     |
| `--max_iter`  | Number of optimization iterations                | 100     |
| `--show_iter` | Print loss every N iterations                    | 5       |

---

## **Output**  
The stylized image is saved at the specified output path and displayed automatically.

---

## **Notes**  
- Uses **VGG19 pre-trained on ImageNet** for feature extraction.  
- Uses **L-BFGS optimizer** for better convergence.  
- The content and style images should be of **similar size** for better results.  

---

🚀 **Enjoy creating AI-powered art!** 🚀

