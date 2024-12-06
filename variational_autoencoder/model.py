import torch

from torch import nn

# class Encoder(nn.Module):
#     """"""

#     def __init__(self, input_dims, hidden_dims, latent_dims):
#         super().__init__()

#         # Convolutional layer
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dims, hidden_dims // 2, 3, stride=1, padding=0),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_dims // 2, hidden_dims // 4, 3, stride=1, padding=0),
#             nn.BatchNorm2d(hidden_dims // 4),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_dims // 4, hidden_dims // 8, 3, stride=1, padding=0),
#             nn.ReLU(True)
#         )

#         # Flatten layer
#         self.flatten = nn.Flatten(start_dim=1)

#         # Fully connected layers
#         self.fully_connected_encoder = nn.Sequential(
#             nn.Linear(hidden_dims // 8 * 10 * 10, 128),
#             nn.ReLU(True),
#             nn.Linear(128, latent_dims)
#         )
#         self.fully_connected_mu = nn.Linear(latent_dims, latent_dims)
#         self.fully_connected_sigma = nn.Linear(latent_dims, latent_dims)

        
#     @staticmethod
#     def reparameterize(mean, log_variance):
#         """Method implements 'reparamaterization trick' for variational autoencoder."""
#         standard_deviation = torch.exp(0.5 * log_variance)
#         # Re-write as function of random variable
#         epsilon = torch.rand_like(standard_deviation)
#         return mean + (epsilon * standard_deviation)
    
#     def forward(self, data):
#         output = self.encoder(data)
#         flattened_output = self.flatten(output)
#         latent_variable = self.fully_connected_encoder(flattened_output)
        
#         # Calculate mean and log variance
#         mu = self.fully_connected_mu(latent_variable)
#         sigma = self.fully_connected_sigma(latent_variable)
#         latent_vector = self.reparameterize(mu, sigma)
        
#         return latent_vector, mu, sigma


# class Decoder(nn.Module):
#     """"""

#     def __init__(self, input_dims, hidden_dims, latent_dims):
#         super().__init__()

#         # Fully connected layers
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(latent_dims, 128),
#             nn.ReLU(True),
#             nn.Linear(128, hidden_dims // 8 * 10 * 10),
#             nn.ReLU(True)
#         )

#         # Unflatten layer
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(hidden_dims // 8, 10, 10))

#         # Convolutional layers
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(hidden_dims // 8, hidden_dims // 4, 3, stride=1, output_padding=0),
#             nn.BatchNorm2d(hidden_dims // 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(hidden_dims // 4, hidden_dims // 2, 3, stride=1, output_padding=0),
#             nn.BatchNorm2d(hidden_dims // 2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(hidden_dims // 2, input_dims, 3, stride=1, output_padding=0),
#         )

#     def forward(self, x):
#         x = self.decoder_fc(x)
#         x = self.unflatten(x)
#         x = self.decoder(x)
#         x = torch.sigmoid(x)
#         return x


# class VariationalAutoencoder(nn.Module):
#     """"""

#     def __init__(self, input_dims, hidden_dims, latent_dims, encoder=Encoder, decoder=Decoder):
#         super().__init__()
#         self.encoder = encoder(input_dims, hidden_dims, latent_dims)
#         self.decoder = decoder(input_dims, hidden_dims, latent_dims)

#     def forward(self, x):
#         z, mu, sigma = self.encoder(x)
#         return self.decoder(z), mu, sigma




class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(Encoder, self).__init__()
        # Convolutional layers
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_dims, 8, 3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 16, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 3, stride=1, padding=0),
        #     nn.ReLU(True)
        # )

        self.x1 = nn.Conv2d(input_dims, 8, 3, stride=1, padding=1)
        self.x2 = nn.ReLU(True)
        self.x3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.x4 = nn.BatchNorm2d(16)
        self.x5 = nn.ReLU(True)
        self.x6 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
        self.x7 = nn.ReLU(True)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layers
        # self.fully_connected_encoder = nn.Sequential(
        #     nn.Linear(32 * 254 * 2046, latent_dims),  # Adjusted based on the new input size
        #     # nn.ReLU(True),
        #     # nn.Linear(128, latent_dims)
        # )
        self.fully_connected_mu = nn.Linear(32 * 254 * 2046, latent_dims)
        self.fully_connected_sigma = nn.Linear(32 * 254 * 2046, latent_dims)

    @staticmethod
    def reparameterize(mean, log_variance):
        """Method implements 'reparameterization trick' for variational autoencoder."""
        standard_deviation = torch.exp(0.5 * log_variance)
        epsilon = torch.rand_like(standard_deviation)
        return mean + (epsilon * standard_deviation)

    def forward(self, data):
        # output = self.encoder(data)
        encoder1 = self.x1(data)
        print(encoder1.shape)
        encoder1 = self.x2(encoder1)
        print(encoder1.shape)
        encoder1 = self.x3(encoder1)
        print(encoder1.shape)
        encoder1 = self.x4(encoder1)
        print(encoder1.shape)
        encoder1 = self.x5(encoder1)
        print(encoder1.shape)
        encoder1 = self.x6(encoder1)
        print(encoder1.shape)
        output = self.x7(encoder1)


        flattened_output = self.flatten(output)
        # latent_variable = self.fully_connected_encoder(flattened_output)

        # Calculate mean and log variance
        mu = self.fully_connected_mu(flattened_output)
        sigma = self.fully_connected_sigma(flattened_output)
        latent_vector = self.reparameterize(mu, sigma)

        return latent_vector, mu, sigma


class Decoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(Decoder, self).__init__()

        # Fully connected layers
        # self.decoder_fc = nn.Sequential(
        #     # nn.Linear(latent_dims, 128),
        #     # nn.ReLU(True),
        #     nn.Linear(latent_dims, 32 * 254 * 2046),  # Adjusted based on the new input size
        #     nn.ReLU(True)
        # )

        self.decoder_fc = nn.Linear(latent_dims, 32 * 254 * 2046)  # Adjusted based on the new input size
        # self.decoder_relu = nn.ReLU(True)

        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 254, 2046))  # Adjusted based on the new input size

        # Convolutional layers
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, input_dims, 3, stride=2, padding=1, output_padding=1),
        # )

        self.conv1 = nn.ConvTranspose2d(32, 16, 3, stride=1, output_padding=0)
        self.conv2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ReLU(True)
        self.conv4 = nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1, output_padding=0)
        self.conv5 = nn.BatchNorm2d(8)
        self.conv6 = nn.ReLU(True)
        self.conv7 = nn.ConvTranspose2d(8, input_dims, 3, stride=1, padding=1, output_padding=0)


    def forward(self, x):
        print("devon")
        # x = self.decoder_fc(x)
        x = self.decoder_fc(x)
        print(x.shape)
        # x = self.decoder_relu(x)
        # print(x.shape)
        x = self.unflatten(x)
        print(x.shape)
        # x = self.decoder(x)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        x = self.conv6(x)
        print(x.shape)
        x = self.conv7(x)
        print(x.shape)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dims, encoder=Encoder, decoder=Decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder(input_dims, latent_dims)
        self.decoder = decoder(input_dims, latent_dims)

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), mu, sigma
    



# class Encoder(nn.Module):
#     def __init__(self, input_dims, latent_dims):
#         super(Encoder, self).__init__()
#         # Convolutional layers
#         # self.encoder = nn.Sequential(
#         #     nn.Conv2d(input_dims, 8, 3, stride=1, padding=1),
#         #     nn.ReLU(True),
#         #     nn.Conv2d(8, 16, 3, stride=1, padding=1),
#         #     nn.BatchNorm2d(16),
#         #     nn.ReLU(True),
#         #     nn.Conv2d(16, 32, 3, stride=1, padding=0),
#         #     nn.ReLU(True)
#         # )

#         self.x1 = nn.Conv2d(input_dims, 8, 9, stride=2, padding=0)
#         self.pool1 = nn.MaxPool2d(3, 1)
#         self.x2 = nn.ReLU(True)
#         self.x3 = nn.Conv2d(8, 16, 5, stride=1, padding=0)
#         self.pool2 = nn.MaxPool2d(3, 2)
#         self.x4 = nn.BatchNorm2d(16)
#         self.x5 = nn.ReLU(True)
#         self.x6 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
#         self.pool3 = nn.MaxPool2d(3, 2)
#         self.x7 = nn.ReLU(True)

#         # Flatten layer
#         self.flatten = nn.Flatten(start_dim=1)

#         # Fully connected layers
#         # self.fully_connected_encoder = nn.Sequential(
#         #     nn.Linear(32 * 254 * 2046, latent_dims),  # Adjusted based on the new input size
#         #     # nn.ReLU(True),
#         #     # nn.Linear(128, latent_dims)
#         # )
#         self.fully_connected_mu = nn.Linear(32 * 27 * 251, latent_dims)
#         self.fully_connected_sigma = nn.Linear(32 * 27 * 251, latent_dims)

#     @staticmethod
#     def reparameterize(mean, log_variance):
#         """Method implements 'reparameterization trick' for variational autoencoder."""
#         standard_deviation = torch.exp(0.5 * log_variance)
#         epsilon = torch.rand_like(standard_deviation)
#         return mean + (epsilon * standard_deviation)

#     def forward(self, data):
#         # output = self.encoder(data)
#         print("encoder")
#         # encoder1 = self.x1(data)
#         # print(encoder1.shape)
#         # encoder1 = self.x2(encoder1)
#         # print(encoder1.shape)
#         # encoder1 = self.x3(encoder1)
#         # print(encoder1.shape)
#         # encoder1 = self.x4(encoder1)
#         # print(encoder1.shape)
#         # encoder1 = self.x5(encoder1)
#         # print(encoder1.shape)
#         # encoder1 = self.x6(encoder1)
#         # print(encoder1.shape)
#         # output = self.x7(encoder1)
#         # print(output.shape)
#         encoder1 = self.x1(data)
#         print(encoder1.shape)
#         pool1 = self.pool1(encoder1)
#         print(pool1.shape)
#         encoder1 = self.x2(pool1)
#         print(encoder1.shape)
#         encoder1 = self.x3(encoder1)
#         print(encoder1.shape)
#         pool2 = self.pool2(encoder1)
#         print(pool2.shape)
#         encoder1 = self.x4(pool2)
#         print(encoder1.shape)
#         encoder1 = self.x5(encoder1)
#         print(encoder1.shape)
#         encoder1 = self.x6(encoder1)
#         print(encoder1.shape)
#         pool3 = self.pool3(encoder1)
#         print(pool3.shape)
#         output = self.x7(pool3)
#         print(output.shape)

#         flattened_output = self.flatten(output)
#         print(flattened_output.shape)
#         # latent_variable = self.fully_connected_encoder(flattened_output)

#         # Calculate mean and log variance
#         mu = self.fully_connected_mu(flattened_output)
#         print(mu.shape)
#         sigma = self.fully_connected_sigma(flattened_output)
#         print(sigma.shape)
#         latent_vector = self.reparameterize(mu, sigma)
#         print(latent_vector.shape)

#         return latent_vector, mu, sigma


# class Decoder(nn.Module):
#     def __init__(self, input_dims, latent_dims):
#         super(Decoder, self).__init__()

#         # Fully connected layers
#         # self.decoder_fc = nn.Sequential(
#         #     # nn.Linear(latent_dims, 128),
#         #     # nn.ReLU(True),
#         #     nn.Linear(latent_dims, 32 * 254 * 2046),  # Adjusted based on the new input size
#         #     nn.ReLU(True)
#         # )

#         self.decoder_fc = nn.Linear(latent_dims, 32 * 27 * 251)  # Adjusted based on the new input size
#         # self.decoder_relu = nn.ReLU(True)

#         # Unflatten layer
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 27, 251))  # Adjusted based on the new input size

#         # Convolutional layers
#         # self.decoder = nn.Sequential(
#         #     nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
#         #     nn.BatchNorm2d(16),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
#         #     nn.BatchNorm2d(8),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(8, input_dims, 3, stride=2, padding=1, output_padding=1),
#         # )

#         self.upsample1 = nn.Upsample(scale_factor=2)
#         self.conv1 = nn.ConvTranspose2d(32, 16, 3, stride=1, padding=2, output_padding=0)
#         self.conv2 = nn.BatchNorm2d(16)
#         self.conv3 = nn.ReLU(True)
#         self.upsamlple2 = nn.Upsample(scale_factor=2)
#         self.conv4 = nn.ConvTranspose2d(16, 8, 5, stride=1, padding=0, output_padding=0)
#         self.conv5 = nn.BatchNorm2d(8)
#         self.conv6 = nn.ReLU(True)
#         self.conv7 = nn.ConvTranspose2d(8, input_dims, 9, stride=2, padding=0, output_padding=0)


#     def forward(self, x):
#         print("decoder")
#         # x = self.decoder_fc(x)
#         x = self.decoder_fc(x)
#         print(x.shape)
#         # x = self.decoder_relu(x)
#         # print(x.shape)
#         x = self.unflatten(x)
#         print(x.shape)
#         # x = self.decoder(x)
#         x = self.upsample1(x)
#         print(x.shape)
#         x = self.conv1(x)
#         print(x.shape)
#         x = self.conv2(x)
#         print(x.shape)
#         x = self.conv3(x)
#         print(x.shape)
#         x = self.upsamlple2(x)
#         print(x.shape)
#         x = self.conv4(x)
#         print(x.shape)
#         x = self.conv5(x)
#         print(x.shape)
#         x = self.conv6(x)
#         print(x.shape)
#         x = self.conv7(x)
#         print(x.shape)
#         x = torch.sigmoid(x)
#         return x


# class VariationalAutoencoder(nn.Module):
#     def __init__(self, input_dims, latent_dims, encoder=Encoder, decoder=Decoder):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = encoder(input_dims, latent_dims)
#         self.decoder = decoder(input_dims, latent_dims)

#     def forward(self, x):
#         z, mu, sigma = self.encoder(x)
#         return self.decoder(z), mu, sigma