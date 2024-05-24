# Arabic-Handwritten-Character-Recognition-with-CNNs

## Arabic Handwritten Character Recognition with CNNs: A Comprehensive Exploration

### Project Overview

This project is dedicated to exploring various techniques and strategies for Arabic Handwritten Character Recognition (AHCR) using Convolutional Neural Networks (CNNs). The main objective is to guide students through the process of building, training, and optimizing a CNN model specifically designed for AHCR. The project encompasses several tasks, each targeting different aspects of CNN architecture and training methodologies, including data augmentation, leveraging pre-established architectures, and employing transfer learning.

### CNN Architecture

**First Convolutional Block (16 filters):**  
The initial convolutional block of the designed CNN features a Conv2D layer employing 16 filters with a 3x3 kernel and 'same' padding. The ReLU activation function is applied to introduce non-linearity, and a MaxPooling2D layer with a 2x2 pool size and a stride of 1 is incorporated for down-sampling. To enhance regularization, a dropout of 20% is implemented.

**Second Convolutional Block (32 filters):**  
In the second convolutional block, the model utilizes a Conv2D layer with 32 filters, a 3x3 kernel, and 'same' padding. The ReLU activation function is applied for non-linearity, followed by a MaxPooling2D layer with a 2x2 pool size and a stride of 1. Regularization is enforced with a 20% dropout rate.

**Third Convolutional Block (64 filters):**  
The third convolutional block involves a Conv2D layer with 64 filters, utilizing a 3x3 kernel and 'same' padding. ReLU activation is applied for non-linearity, followed by a MaxPooling2D layer with a 2x2 pool size and a stride of 1. A dropout rate of 20% is employed for regularization.

**Fourth Convolutional Block (128 filters):**  
The fourth convolutional block employs a Conv2D layer with 128 filters, a 3x3 kernel, and 'same' padding. ReLU activation is applied, and a MaxPooling2D layer with a 2x2 pool size and a stride of 1 is utilized for down-sampling. Regularization is maintained with a dropout rate of 20%.

**Flatten and Dense Layers:**  
Following the convolutional blocks, the model includes a Flatten layer to convert the output of the last pooling layer into a flat vector. A first Dense layer with 512 neurons and ReLU activation is implemented, and a dropout of 40% is applied for regularization.

**Output Layer:**  
The final layer serves as the output layer for character recognition. It consists of a Dense layer with a number of neurons equal to the specified num_classes. The softmax activation function is employed to facilitate multi-class classification.

**Training Configuration:**  
The training configuration includes the use of the Adam optimizer with a learning rate of 0.001. The chosen loss function is Categorical Crossentropy, suitable for multi-class classification tasks. The model's performance is evaluated based on accuracy.

### Conclusion and Results

**Task 1: Building and Training a Custom CNN Network**  
In our first task, we built a custom Convolutional Neural Network (CNN) from scratch. This network was carefully structured, layer by layer, to process and learn from the dataset's intricacies. The initial results were promising, with the model achieving a test accuracy of 91.43%. It was a solid foundation, but there was room for improvement.

**Task 2: Data Augmentation**  
Acknowledging the limitation of our dataset's size and diversity, we applied data augmentation techniques to artificially expand our training data. By introducing variations such as rotations, shifts, and zooms, the model was exposed to a broader spectrum of data. This exposure paid off, elevating the test accuracy to 93.75%. The model became more versatile and less prone to overfitting, demonstrating improved generalization on unseen data.

**Task 3: Leveraging Pre-established Architectures (AlexNet)**  
Next, we explored the power of pre-established architectures by implementing AlexNet, a well-known CNN architecture. This move was strategic, aiming to benefit from the architectural insights and proven efficiency of existing models. The results were impressive, as the test accuracy further climbed to 89.43%. This approach combined the robustness of AlexNet with the specificity of our dataset, leading to a high-performing model.

**Task 4: Employing Transfer Learning**  
Finally, we delved into transfer learning, a technique that leverages pre-trained models to accelerate the training process and improve performance. By fine-tuning a model pre-trained on similar tasks, we harnessed the knowledge it had already acquired. This strategy was fruitful, with the test accuracy reaching 87.44%. Transfer learning proved to be a time-efficient and effective method to further refine our model's performance.

**In conclusion**, we pursued four distinct tasks to identify the most effective approach for automatic recognition of Arabic handwritten characters. The first task involved creating and training a custom network, providing a baseline for comparison. The second task involved data augmentation, which enhanced the network's learning and accuracy. The third task utilized the well-established AlexNet architecture, and the fourth task employed transfer learning to adapt a pre-trained model for our specific problem. Each task highlighted the strengths and weaknesses of different methodologies, offering valuable insights into the automatic recognition of Arabic handwritten characters.
