#include<iostream>
#include<utils.h>

int main() {
    std::cout << "Image Preview Test" << std::endl;

    torch::Tensor images = load_tensor("/Users/sinngamkhaidem/Developer/torch_ray/data/images.pt");
    torch::Tensor img = images[50];

    save_image(img, "preview.png");


}