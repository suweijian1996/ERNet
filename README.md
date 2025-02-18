# Infrared and Visible Image Fusion based on Adversarial Feature Extraction and Stable Image Reconstruction (TIM2022)

This project implements an image fusion method that combines infrared and visible images using adversarial feature extraction and stable image reconstruction. The method aims to enhance the quality and robustness of the fused images, making it suitable for various applications in image processing and computer vision.


### Update log
2024/09/04 Uploaded training parameters.

## Getting Started

### Prerequisites

- Python 3.x
- YOLO framework (Ensure that you have YOLO set up)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suweijian1996/ERNet.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the necessary directories and prepare the datasets:
   ```bash
   python ./ERNet/createpath.py
   ```

### Project Structure

- `./ERNet/Fusion_Net/`: Contains the core implementation of the fusion network.
- `./ERNet/createpath.py`: A script to generate the directory structure for training or testing datasets.

### Usage

To train or test the model, navigate to the `./ERNet/Fusion_Net/train_endecoder.py` or `./ERNet/Fusion_Net/fuseimage.py` folder and follow the instructions provided in the corresponding scripts.

### Datasets

If you require the fusion results of ERNet on specific datasets, please send the source images to [su_weijian1996@163.com](mailto:su_weijian1996@163.com) with the note "ERNet".

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Acknowledgements

We acknowledge the contributions of the YOLO framework and other open-source libraries used in this project.

### References

W. Su, Y. Huang, Q. Li, F. Zuo and L. Liu, "Infrared and Visible Image Fusion Based on Adversarial Feature Extraction and Stable Image Reconstruction," in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-14, 2022, Art no. 2510214, doi: 10.1109/TIM.2022.3177717.

---

Feel free to modify any sections to better fit your project's needs!
