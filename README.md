
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
