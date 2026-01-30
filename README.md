
<p align="center">
<img width="200" height="200" alt="snake_icon" src="https://github.com/user-attachments/assets/63f262e8-7556-406e-9913-e58d77085565" />
</p>

# Snake-RL

In this project we've trained 3 agents, namely: **BlindAgent**, **LidarAgent**, **AtariAgent**.
Each of them has its own state-space representation and
all of them are trained using the DQN algoritm (more info at this [link](https://www.nature.com/articles/nature14236)).
After identifying the best state-space representation, the **CerberusAgent** was built by ensembling three pre-trained agents
(3 AtariAgents). You can find more information in the `report.pdf`.

In the following we show the record achieved by each agent.

---

<div align="center">

<table>
  <tr>
    <th align="center">CerberusAgent</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e770b9af-a43c-4857-95ce-0a957af7beac">
    </td>
  </tr>
</table>
  
<table>
  <tr>
    <th align="center">BlindAgent</th>
    <th align="center">LidarAgent</th>
    <th align="center">AtariAgent</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9a6f0f67-dcc4-4b08-8ff4-f094c3772954">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fcd08280-450c-4e6e-a022-6b024721e5e6">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/747d96d0-5176-4562-b8cf-2bd987cca479">
    </td>
  </tr>
</table>
  
</div>

---

## AtariAgent

I will not introduce all the agents since they're explained in the `report.pdf`, in this readme i will present only the
**CerberusAgent**. However, to introduce it, I need to present the **AtariAgent**.

The AtariAgent takes its name from the famous [DeepMind paper](https://www.nature.com/articles/nature14236).
In the paper they use a CNN and the input is built by stacking the last 4 frames of the game, the
AtariAgent use the same concept. Its Q-network has the following structure:

<div align="center">
  <img width="2856" align="center" height="432" alt="image" src="https://github.com/user-attachments/assets/386ab609-56c2-4b84-a0ce-781e2f788d68" />
</div>

## CerberusAgent

### Why using it?

Is AtariAgent not enough to win Snake? No. Well, at least I was't able to train it to do so. By looking at the record achieved by
the AtariAgent you will see that the policy learned can be splitted in two: at the beginning the snake rushes towards the
food (like BlindAgent does), then it switches to a more survival policy and it starts to follow a circular path.

So we can just train 2 different AtariAgent: the first one is the one we have already trained (aka the one playing in
the gif); the second one is trained using a longer snake (like 50 for example) from the beginning. In this way the second one
must learn a survival strategy and by combining the two agents we should win the game.

### Why 3 heads and not just 2?

The CerberusAgent comes from that observation. This agent is composed by 3 AtariAgents, we call them **heads**. **Why 3 and not just 2?** The short answer is: *because it
works better in this way*. The idea behind the introduction of the second head, is related to the input distribution of the third head. However note that, by removing it, you
will get similar results.

### How does it play?

At each step of the game, a single head gives the next action to perform. The head that decides the action is selected based on the current score.
The simple logic is reported in the following picture:

<div align="center">
<img width="500" height="260" alt="image" src="https://github.com/user-attachments/assets/9093ed5a-41b7-473b-a895-506f61aa16ac" />
</div>

### Does it always win the game?

Hell no. The overall strategy *can* win the game but it is far from beeing a perfect strategy.

<div align="center">
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/d061bb84-6581-4ff7-8700-7050b6fd8b2e" />
</div>

In the picture above we can see the score distribution. The bars are colored according to the head that was responsible for the snake’s death.
Note that: when game starts, the lenght of the snake is 3 and the maximum is 100, so the maximum score is 97 (the score is the number of apples eaten). 
We observe that the agent reaches the end-game very frequently, although it is less likely to fully win the game.
