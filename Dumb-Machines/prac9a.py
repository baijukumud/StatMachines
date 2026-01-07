import random
import matplotlib.pyplot as plt

# Real data distribution: mean = 4
def sample_real():
    return random.gauss(4, 1)

# Generator: starts with noise and learns a shift
class Generator:
    def __init__(self):
    self.bias = random.uniform(-1, 1)  # this is the "parameter" it learns
    
    def generate(self):
        noise = random.uniform(-1, 1)
        return noise + self.bias

    def update(self, gradient, lr=0.01):
        self.bias += lr * gradient

# Discriminator: simple linear classifier
class Discriminator:
    def __init__(self):
        self.threshold = 0.0  # threshold to separate real and fake
    
    def predict(self, x):
        return 1 if x > self.threshold else 0  # 1 = real, 0 = fake

    def update(self, real_samples, fake_samples):
        real_mean = sum(real_samples) / len(real_samples)
        fake_mean = sum(fake_samples) / len(fake_samples)
        self.threshold = (real_mean + fake_mean) / 2  # separate boundary

# Instantiate
G = Generator()
D = Discriminator()

# Training loop
for epoch in range(500):
    real_data = [sample_real() for _ in range(10)]
    fake_data = [G.generate() for _ in range(10)]

    # Update Discriminator
    D.update(real_data, fake_data)

    # Generator loss: wants D(fake) = 1
    grad = 0.0
    for x in fake_data:
        prediction = D.predict(x)
        grad += (1 - prediction) # if D says fake (0), G wants to improve
    
    grad /= len(fake_data)
    G.update(grad)
    
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch} | Generator bias: {G.bias:.4f} | "
            f"Discriminator threshold: {D.threshold:.4f}"
        )

# Plot results
generated_samples = [G.generate() for _ in range(1000)]
real_samples = [sample_real() for _ in range(1000)]

plt.hist(real_samples, bins=40, alpha=0.5, label="Real")
plt.hist(generated_samples, bins=40, alpha=0.5, label="Generated")
plt.legend()
plt.title("Generated vs Real Data (Pure Python GAN)")
plt.show()
