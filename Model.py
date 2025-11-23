import tensorflow as tf
import numpy as np


def Linear_QNet(input_size, hidden_size, output_size):
    inputs = tf.keras.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(units=hidden_size, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(units=output_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )


    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=float)
        next_state = np.array(next_state, dtype=float)
        action = np.array(action, dtype=float)
        reward = np.array(reward, dtype=float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model.predict(state, verbose=0)
        #print(pred.shape)
        #print(pred)

        target = np.copy(pred)
        next_pred = self.model(next_state)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * np.max(next_pred[idx])

            target[idx, np.argmax(action[idx]).item()] = Q_new

        #print("Pred:\n", pred)
        #print("Target:\n", target)
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.model.train_on_batch(state, target)
        #self.optimizer.zero_grad()
        #loss = self.criterion(target, pred)
        #loss.backward()
        #self.optimizer.step()


