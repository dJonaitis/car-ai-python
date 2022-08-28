# car-ai-python-
Race cars that learn to drive around a course in Python, made using the NEAT-Python library. The cars procedurally learn over time to be better and better using the neuroevolution of augmenting topologies (NEAT) algorithm. All art is done by me.

## Usage
1. Ensure you have the prerequisite dependencies (pygame, NEAT-Python).
2. To install the libraries use the terminal commands: ``$ pip install neat-python`` and ``$ pip install pygame``.
3. Run ``main.py`` and the window will open.
4. The cars will begin driving on their own.
5. The current generation and remaining cars in the generation can be seen in the top left.
6. Click on the button in the top right to toggle the radars on or off. Radars are the lines in front of each car that denote how far away the end of the track is (imagine it as the vision of the car).

## Uploading other tracks
Any track that works on the basic premise of being completely surrounded by a singular color will work. Currently the color is ``(46, 114, 46)`` in RGB.  
  
To change the color, simply change the RGB values in the ``grassColor`` variable at the top of ``main.py``.  
  
The only requirements for a track is that the road upon which the cars drive is surrounded by ``grassColor`` and it loops.

## Changing the amount of cars
To increase/decrease the amount of cars in each generation, you must do the following:  
1. Change ``startingCars`` in ``main.py`` to the desired amount
2. Change ``pop_size`` in ``config.txt`` to the desired amount

Increasing the amount of cars will have noticeable effects on performance depending on your system. If you have a lower end system and are experiencing performance issues, consider lowering the amount of cars.

## Further reading
If you want to learn more about NEAT, consider the following sources.
1. K. O. Stanley and R. Miikkulainen, "[Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)" This is the original paper detailed by the developer of the algorithm.
2. [NEAT-Python Documentation](https://neat-python.readthedocs.io/en/latest/)
3. [Wikipedia article on NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
4. [An interesting video about NEAT by ForrestKnight](https://www.youtube.com/watch?v=5RR1T_-zVws)
