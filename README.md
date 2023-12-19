# Aspiration

## A research project with the goal of building AI agents with the following properties:
### 1. Real Time - the ability to learn continuously online, without static batch training
### 2. State Action Mapping - the ability navigate any environment within limits of its available actions, via approximating the function that maps a state and action to the following state
### 3. State Aspiring - the ability to generate actions based on aspiring for better states, vis approximating the function that maps a state and a better next state to the action that achieves that better state.

## Sn+1 = State_Predictor ( Sn, An )
Next token prediction has shown to produce capable emergent behaviors. Transformer based LLM's can seemingly interpolate the space of written knowlege, and therefore sample a new idea that was not originally in the dataset, but rather inbetween data points on a continuous space of concepts encoded in text. I propose a succesfull approximation of a state prediction function will allow an agent to traverse the dynamics of any environment.
### Training
Ensure Sn = SP ( Sn-1, An-1 ). Retain previous states and actions in memory, and train the model to correctly output the current state.

## An = Action_Generator ( Sn, Sn+1 )
The accuracy of approximating the state prediction function is limited to the space of prior experience. The SP model is not reliably accurate for state action inputs it has not already recorded the outcomes of. To combat situations where the model refuses to try new actions, and gets stuck in a loop, this model can help by allowing the agent to 'dream' or 'aspire' a state it has never experienced, and it will result in an action most probable to achieve that state, even if that action has never been taken.
