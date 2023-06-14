# DDQN AI Super Mario Bros
 This is the base for the DDQN agent for the Super Mario Bros
 
 To run this project: 
 
 First go the the requirements.txt file and make sure you have pip installed all the required packages. 
 
 Then, go the train.py and modify the training parameters in the parse_opt function. 
 
 Then run the main part, and the result will be stored in the run folder as a new run.
 
 File Explanation:
 
 The agent.py contains the DDQN model for the super mario bros AI. 
 
 The network.py contains the self-implemented 3-layer Conv2d network .
 
 The super_mario_env.py file contains the environment reduction simplification for the network.
 
 The train.py contains the training process with logging and parsing.
 
 The requirements.txt contains all the software needed for this project.
 
 The runs folder contains all the previous trainings/testings.
