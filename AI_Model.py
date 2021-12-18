from plot_results import plot
from saving_model import save_model
from loading_model import load_model
from train_neural_network import train
from neural_network_model import neural_network
from creating_training_val_test_datasets import create_datasets


def build_AI_model(dataset, channel, width, height):
    dataset_params = create_datasets(dataset, width, height)
    train_loader, val_loader, test_loader, test_batch_loader, train_dataset, val_dataset, test_dataset = dataset_params

    model, train_step, loss_fn = neural_network(channel, width, height)

    train_results = train(model, train_loader, train_dataset, val_loader, val_dataset, train_step, loss_fn, width)
    training_accuracies, training_losses, validation_accuracies, validation_losses = train_results

    save_model(model)

    # Loading previous ready-trained models
    model = load_model()

    plot(model, width, height, train_loader, train_dataset, training_accuracies, training_losses, validation_accuracies,
         validation_losses, test_dataset, test_loader, test_batch_loader)

    """
    indices_index = 0
    for image, _ in test_loader:
        # image_path = './Cars License Plates/' + row[0]
        # image: Image = Image.open(image_path)
        # image_tensor: transforms = transform(image)
        # y_hat = model.predict(image.reshape(1, WIDTH, HEIGHT, CHANNEL)).reshape(-1) * WIDTH
        # y_hat = model(image_tensor)
        y_hat = model(image)

        xt, yt = int(y_hat[0][0] * WIDTH), int(y_hat[0][1] * WIDTH)
        xb, yb = int(y_hat[0][2] * WIDTH), int(y_hat[0][3] * WIDTH)

        # image = cv2.resize(cv2.imread("./Cars License Plates/" + row[0]) / 255.0, dsize=(WIDTH, HEIGHT))
        # image = cv2.cvtColor(image.detach().numpy(), cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
        image_index = test_dataset.indices[indices_index]
        image_path = fr'./Dataset/Cars License Plates/Car_License_Image_{image_index}.jpeg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
        image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 0, 255), 1)
        plt.imshow(image)
        plt.show()

        indices_index += 1
    """

    """
    for image_index in test_dataset.indices:
    image_path = fr'./Dataset/Cars License Plates/Car_License_Image_{image_index}.jpeg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    y_hat = model(torch.tensor(image))

    xt, yt = y_hat[0][0], y_hat[0][1]
    xb, yb = y_hat[0][2], y_hat[0][3]

    # Adding a rectangle at the corresponding points according to the dataset
    image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()
    """
