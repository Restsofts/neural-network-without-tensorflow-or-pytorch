import numpy as np
import matplotlib.pyplot as plt

# Fixing the random seed for reproducibility
np.random.seed(42)

# دوال التفعيل ومشتقاتها


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid_derivative(x):
    return x * (1 - x)

# Xavier initialization for better weight scaling


def xavier_initialization(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)

# التفعيل الأمامي


def forward(X, W1, W2):
    z1 = np.dot(X, W1)
    a1 = tanh(z1)
    z2 = np.dot(a1, W2)
    a2 = sigmoid(z2)
    return a2, a1

# الانتشار العكسي


def backward(X, y, W1, W2, a1, a2, learning_rate):
    error_output = y - a2
    delta_output = error_output * sigmoid_derivative(a2)

    error_hidden = delta_output.dot(W2.T)
    delta_hidden = error_hidden * tanh_derivative(a1)

    W2 += a1.T.dot(delta_output) * learning_rate
    W1 += X.T.dot(delta_hidden) * learning_rate

    return W1, W2


# إعداد البيانات
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# إعداد الأوزان باستخدام Xavier initialization
input_size = 2
hidden_size = 4  # يمكن تجربة 6 أو 8 لزيادة تعقيد الشبكة
output_size = 1
W1 = xavier_initialization(input_size, hidden_size)
W2 = xavier_initialization(hidden_size, output_size)

# التدريب
epochs = 20000
learning_rate = 0.01
losses = []

for epoch in range(epochs):
    a2, a1 = forward(X, W1, W2)
    W1, W2 = backward(X, y, W1, W2, a1, a2, learning_rate)

    loss = np.mean(np.square(y - a2))
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# تقييم أداء الشبكة
predictions = np.round(a2)
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy * 100}%')

# عرض الخسارة
plt.plot(losses)
plt.title('Loss over time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
