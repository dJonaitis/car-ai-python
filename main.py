import os
import sys
import math
import random

import neat
import pygame

pygame.init()
clock = pygame.time.Clock()

grassColor = pygame.Color(46, 114, 46) # color for grass
startingCars = 20 # the amount of cars in the start of the game, ensure that this value is the same as pop_size in config.txt
carVarCount = 5 # the amount of car variations

width, height = 1244, 1016
window = pygame.display.set_mode((width, height))

#loading assets

trackSprite = pygame.image.load('assets/track.png').convert()
toggleOn = pygame.image.load('assets/toggleOn.png').convert_alpha()
toggleOff = pygame.image.load('assets/toggleOff.png').convert_alpha()
toggleSprites = [toggleOn, toggleOff]
fontObj = pygame.font.Font('assets/8bit.ttf', 30)

class Car(pygame.sprite.Sprite):
    def __init__(self, spriteIndex):
        super().__init__()
        self.spriteIndex = spriteIndex
        self.originalSprite = pygame.image.load(os.path.join("assets", f"car{self.spriteIndex}.png"))
        self.image = self.originalSprite
        self.rect = self.image.get_rect(center=(490, 820))

        self.velocity = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotationalVelocity = 5
        self.direction = 0 # -1 when turning left, +1 when turning right, when straight 0

        self.alive = True
        self.radars = []
    
    def update(self):
        self.radars.clear()

        # driving
        self.rect.center += self.velocity * 6
        
        # turning
        if self.direction == 1: # turning right
            self.angle -= self.rotationalVelocity
            self.velocity.rotate_ip(self.rotationalVelocity)
        
        if self.direction == -1: # turning left
            self.angle += self.rotationalVelocity
            self.velocity.rotate_ip(-self.rotationalVelocity)

        self.image = pygame.transform.rotozoom(self.originalSprite, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)
        
        for angle in (-60, -30, 0, 30, 60):
            self.radar(angle)
        
        self.checkCollision()
        self.getData()
    
    def radar(self, angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not window.get_at((x, y)) == grassColor and length < 200: # color of the grass
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + angle)) * length)

        if toggle:
            pygame.draw.line(window, (255, 255, 255), self.rect.center, (x, y), 1)
            pygame.draw.circle(window, (255, 0, 0 ), (x, y), 4)

        distance = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2))) # distance between center of car and tip of radar
        self.radars.append([angle, distance])
    
    def checkCollision(self):
        length = 40
        colPointR = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length), int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        colPointL = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length), int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        
        if window.get_at(colPointR) == grassColor or window.get_at(colPointL) == grassColor:
            self.alive = False
    
    def getData(self):
        input = [0, 0, 0, 0, 0]

        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])

        return input 
    

def kill(i):
    cars.pop(i)
    genome.pop(i)
    networks.pop(i)

genCount = 0 


def evaluateGenomes(genomes, config):
    global cars, genome, networks, toggleIndex, toggle, genCount, remainingCars

    remainingCars = startingCars
    cars = []
    genome = []
    networks = []
    toggle = True
    toggleIndex = 0
    genCount += 1

    for genome_id, ge in genomes:
        carIndex = random.randrange(1, carVarCount+1, 1)
        cars.append(pygame.sprite.GroupSingle(Car(carIndex)))
        genome.append(ge)
        network = neat.nn.FeedForwardNetwork.create(ge, config)
        networks.append(network)
        ge.fitness = 0
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if pygame.mouse.get_pressed()[0]:
                position = pygame.mouse.get_pos()
                if position[0] >= 935 and position[1] <= 250 and position[1] >= 100 and position[0] <= 1235:
                    toggle = not toggle
                    if toggle:
                        toggleIndex = 0
                    else:
                        toggleIndex = 1
            

        window.blit(trackSprite, (0, 0))
        

        if len(cars) == 0: # end while loop when all cars are dead
            break

        for i, car in enumerate(cars):
            genome[i].fitness += 1

            if not car.sprite.alive:
                kill(i)
                remainingCars -= 1

        for i, car in enumerate(cars):
            output = networks[i].activate(car.sprite.getData())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        for car in cars:
            car.draw(window)
            car.update()
        
        generationContent = f"Gen: {genCount}"
        carContent = f"Cars: {remainingCars}/{startingCars}"
        genText = fontObj.render(str(generationContent), True, (0, 0, 0))
        carText = fontObj.render(str(carContent), True, (0, 0, 0))
        window.blit(genText, (50, 100))
        window.blit(carText, (200, 100))
        window.blit(toggleSprites[toggleIndex], (935, 100))
        pygame.display.update()
        clock.tick(120)



def run(path):
    global population

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)

    population.run(evaluateGenomes, startingCars)

if __name__ == '__main__':
    localDir = os.path.dirname(__file__) 
    configPath = os.path.join(localDir, 'config.txt')
    run(configPath)