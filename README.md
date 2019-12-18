# multimodal_conversational_analysis
Signal processing course project! In this project, we aim to produce a model on conversational sentiment and emotion analysis that uses visual features in addition to the audio and textual features previously used. The goal of the project was to produce a novel model that (hopefully) achieves better results than the previously published works. While this goal was not achieve, substantial exploration of novel architectural additions was performed, leading to a good guideline for future research.

Pytorch Facenet source: https://github.com/timesler/facenet-pytorch

Frame Attention Network source: https://github.com/Open-Debin/Emotion-FAN

# Milestones
- We're currently in the process of running the first round of experiments on the baselines. The lack of availability of public repositories for the SOTA paper, namely DialogueGCN, has slowed us down as we're implementing their (quite large and detailed) methods from scratch. That said our pre-processing code is ready (and tested using a dummy model), and we're very close to finishing up on the bugs on the model. We expect to have the experiments be running by next week.
- We've already done our research on using attention masks to visually use the videos in the model and have already formulated the algorithmic extension to DialogueGCN. The bottleneck is again finishing up implementing their base model.
- We've evaluated the papers provided under Literature in this repository.
- While we're slightly slowed down by the upcoming CVPR deadline, and as a result, not quite where we wanted to be given the fact that we look at this project as more than just a project and something that could potentially turn into a paper, we expect to regain substantial speed after the deadline.

# Finished Product:
- This project is complete. Report has been submitted!
