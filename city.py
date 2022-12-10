import numpy as np
import cv2

import math
import random
from random import choice


import gym
from gym import Env, spaces

#### COLORS (BGR) ####
BLUE=  [255, 0,   0]
GREEN= [0,   255, 0]
RED=   [0,   0,   255]
WHITE= [255, 255, 255]
GREY=  [240, 240, 240]
BLACK= [0,   0,   0]
#### BUILDINGS #######
WASTELAND = 0
OFFICE    = 1
HOUSE     = 2
PARK      = 3
COM       = 4
######################

class City(Env):
    def __init__(self, mapshape = (10, 10), start_shape = (3, 3),path='./'):
        super(City, self).__init__()
        self.path=path
        #### reset le programme quand il reste STOP cases vide
        self.stop = max(mapshape)*4
        #on initialise la map memorisant lenvironement
        self.mapshape = mapshape
        #on defini la taille de lespace d observation et lespace
        self.observation_shape = (7,7)
        self.observation_space = spaces.Box(low=0, high=5, shape=self.observation_shape, dtype = int)
        #on defini lespace de depart qui sera generer aleatoirement
        self.start_shape = start_shape
        self.start_size = start_shape[0]
        #on defini la taille de limage qui representera l environement
        self.canvas_shape = 700, 700, 3 # width, height, color (BGR)
        #on defini la variable qui representera notre environement
        self.canvas = np.ones(self.canvas_shape, dtype = np.uint8) * 0
        #on defini le nombre daction (ici 4)
        self.action_space = spaces.Discrete(4)
        #on initilise la sum de la reward comuler sur les episodes.
        self.sum=0
        pass
    #fonction de reset de levironement
    def reset(self, random_start = True):
        #on sauvegarde la sum des reward cumuler
        with open(str(self.path)+str('rewardDQN.txt'), "a+") as fhandle:
            fhandle.write(str(self.sum)+'\n')
        #on reinitialiser la somme des reward
        self.sum=0
        # reset the player's position in the middle of the map
        self.position = (self.mapshape[0] // 2), (self.mapshape[1] // 2)
        # reset the maps
        self.map = np.ones(self.mapshape, dtype = np.uint8) * WASTELAND
        self.offices = []
        self.houses = []
        self.parks = []
        self.coms = []
        self.adjacents_cells = {}
        #on initialise certaine variable
        self.loop_number = 0
        self.start_size = self.start_shape[0]
        self.actual_size = 0
        self.actual_region = 0
        
        # (re)place random houses and offices in the middle of the map
        start_shape = self.start_shape
        if random_start : 
            maisonX=([i for i in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2) if i !=(self.mapshape[1] // 2) ])
            maisonY=choice([i for i in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2) if i != (self.mapshape[0] // 2)])
            oficeX=choice([i for i in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2) if (i != maisonX) and (i!=(self.mapshape[1] // 2))])
            oficeY=choice([i for i in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2) if (i != maisonY) and (i!=(self.mapshape[0] // 2))])
            for y in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2):
                for x in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2):
                    g=random.randrange(2) + 1
                    if (y!=(self.mapshape[0] // 2) or (x!=(self.mapshape[1] // 2))):
                        self.map[y, x] = g
                        if x==maisonX and y==maisonY:
                            self.map[y, x] = HOUSE
                            self.houses.append((y, x))
                        elif x==oficeX and y==oficeY:
                            self.map[y, x] = OFFICE
                            self.offices.append((y, x))
                        elif self.map[y, x] == OFFICE : self.offices.append((y, x))
                        elif self.map[y, x] == HOUSE  : self.houses.append((y, x))
            
                        self.delete_cell((y, x))
                        self.mark_adjacents_cells((y, x))
        x0 = self.position[0]
        y0 = self.position[1]
        OBSMAP = self.getMap(self.map,x0,y0)
        self.vus=OBSMAP
        return OBSMAP

    #fonction qui depuis lenvironement et la position de lagent return lespace d observation assosier.
    def getMap(self,mape,x,y):
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        mape=np.pad(mape, 3, pad_with, padder=0)
        x=x+3
        y=y+3
        return mape[x-3:x+4,y-3:y+4]    
    
    #fonction return la distance au bureau le plus proche, avec une valeur maximum de 3 #on estime que apres la distance a bc moin dinportance relative
    def __search_nearest_office(self, position):
        a=[math.dist(position, office) for office in self.offices]
        if 0 in a:
            a.remove(0)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)
        

    #fonction qui retour la distance au park le plus proche, avec une valeur maximum de 3(permet deviter les bug de distance infini si il ny en a pas et aussi car sinon il ne le verai pas car pas dans espace dobservation)
    def __search_nearest_Park(self, position):
        a=[math.dist(position, office) for office in self.parks]
        if 0 in a:
            a.remove(0)
        a.append(3)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)
    
    #fonction return la distance au bureau le plus proche, avec une valeur maximum de 3
    def __search_nearest_house(self, position):
        a=[math.dist(position, house) for house in self.houses]
        if 0 in a:
            a.remove(0)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)

    #return la distance moyen dun type de batiment avec le centre de la map en inpute, avec une valeur maxime de 3
    def __meanDistance(self,maap,Type):
        coord=np.argwhere(np.array(maap) == Type)
        if len(coord)>0:
            return np.mean(np.array([math.dist((2,2), truc) for truc in coord]))
        else:
            return 1000
    
    #fonction qui renvoi -4 si il y a plus de 30% de bureau sur les case non vide sinon 4 dans le rayon de consideration(5/5)
    def __isTooMutchOffice(self,maap,Type):
        coord=len(np.argwhere(np.array(maap) == Type))
        coord0=len(np.argwhere(np.array(maap) == 0))
        if coord>(int((25-coord0)*0.3)+1):
            return 4
        else:
            return -4
    
    #return le nombre de maison adjasante
    def __NbmaisonAdjasante(self,maap):
        maape=np.array(maap)
        maape=maape[1:4,1:4]
        coord=np.argwhere(np.array(maape) == HOUSE)
        return len(coord)
      
    #return le nombre de maison dans un bufer de 2
    def __NbmaisonAdjasantePARK(self,maap):
        maape=np.array(maap)
        coord=np.argwhere(np.array(maape) == HOUSE)
        return len(coord) 

    #test if a position is occupied
    def __is_free(self, position):
        return self.map[position] == WASTELAND
    #suprime une cellule de la list des cellul adjasante
    def delete_cell(self, position):
        try : del self.adjacents_cells[position]
        except KeyError : pass
    #?
    def mark_cell(self, position):
        y, x = position
        if x < 0 or x >= self.mapshape[0] : return
        if y < 0 or y >= self.mapshape[1] : return
        if tuple(position) in self.houses : return
        if tuple(position) in self.offices : return
        if tuple(position) in self.parks : return
        if tuple(position) in self.coms : return
        try :
            self.adjacents_cells[tuple(position)] += 1
        except KeyError :
            self.adjacents_cells[tuple(position)] = 1
    #?
    def mark_adjacents_cells(self, position):
        y, x = position
        for position in [[y - 1, x - 1], [y - 1, x], [y - 1, x + 1], [y, x - 1], [y, x + 1], [y + 1, x - 1], [y + 1, x], [y + 1, x + 1]] :
            self.mark_cell(position)
        pass

    #retour la reward assosier a une position , un inpute et un batiment. si evaluation est negative mes a jour les memoir
    def q__place(self,position,inpute, is_placing_house,evaluation):
        #on prend une zone plus proche pour evaluer la reward
        inpute=inpute[3-2:3+3,3-2:3+3]

        if is_placing_house==1:#maison
            r1 = self.__search_nearest_office(position)
            r1=1/(math.sqrt(r1**2))
            r2=self.__NbmaisonAdjasante(inpute)
            r2=r2/8
            r3=self.__meanDistance(inpute,COM)
            r3=1/(math.sqrt(r3**2))
            r5=self.__search_nearest_Park(position)
            r5=(1/r5)
            reward =(5*r1+r2+7*r3+7*r5)
            if not evaluation:
                self.houses.append(self.position)
                self.map[self.position] = HOUSE            
            
        elif is_placing_house==0 :#office  
            r1=self.__meanDistance(inpute,HOUSE)
            r1=1/(math.sqrt(r1**2))  
            r2 = self.__search_nearest_office(position)
            r2=1/(math.sqrt(r2**2))
            div=self.__isTooMutchOffice(inpute,OFFICE)
            reward =(5*r1+r2)-(div)
            if not evaluation:    
                self.offices.append(self.position)
                self.map[self.position] = OFFICE    

        elif is_placing_house==2:#park
            r1=self.__NbmaisonAdjasantePARK(inpute)
            reward=r1/5
            if not evaluation:
                self.parks.append(self.position)
                self.map[self.position] = PARK
                
        elif is_placing_house==3:#comercial
            r1=self.__NbmaisonAdjasante(inpute)
            reward=r1/1    
            if not evaluation:
                self.coms.append(self.position)
                self.map[self.position] = COM
                
        else:
            reward=0  
            
        if not evaluation:
            self.delete_cell(self.position)
            self.mark_adjacents_cells(self.position)
            self.sum=self.sum+reward    
        
        return reward

    def GetValue(self):
        array=0
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        for x in range(max(self.mapshape)):
            for y in range(max(self.mapshape)):
                inpute=np.pad(self.map, 3, pad_with, padder=0)
                x1=x+3
                y1=y+3
                inpute=inpute[x1-3:x1+4,y1-3:y1+4]
                inpute[3,3]=0
                value=self.q__place((x,y),inpute, (self.map[x,y])-1,True)
                array=array+value
        #on sauvegarde        
        with open(str(self.path)+str('rewardTotalDQN.txt'), "a+") as fhandle:
            fhandle.write(str(array)+'\n')       
            
    
    # test if a position if out of bound
    def __is_oob(self, position):
        return not(0 <= position[0] < self.mapshape[0]) \
            or not(0 <= position[1] < self.mapshape[1])
    #?
    def select_random_cell(self):
        for position in self.adjacents_cells :
            if self.adjacents_cells[position] >= 2 :
                return position
        return self.adjacents_cells[0]
    #?
    def select_next_cell(self):
        start_position = (self.mapshape[1] - self.start_size) // 2, (self.mapshape[0] - self.start_size) // 2
        start_position = start_position[0] - self.actual_size, start_position[1] - self.actual_size
        
        if self.actual_region == 0 :
            position = start_position[0] - 1,                                           start_position[1]
            position = position[0],                                                     position[1] + self.loop_number
        elif self.actual_region == 1 :
            position = start_position[0],                                               start_position[1] + self.start_size + self.actual_size * 2
            position = position[0] + self.loop_number,                                  position[1]
        elif self.actual_region == 2 :
            position = start_position[0] + self.start_size + self.actual_size * 2,      start_position[1] - 1 + self.start_size  + self.actual_size * 2
            position = position[0],                                                     position[1] - self.loop_number
        elif self.actual_region == 3 :
            position = start_position[0] - 1 + self.start_size + self.actual_size * 2,  start_position[1] - 1
            position = position[0] - self.loop_number,                                  position[1]
                
        self.loop_number += 1
        if self.loop_number > self.start_size + self.actual_size * 2 :
            self.loop_number = 0
            
            self.actual_region += 1
            if self.actual_region >= 4 :
                self.actual_region = 0
                self.actual_size += 1
                
        return position
    #la fonction apeller pr chaque action
    def step(self, action):
        reward = self.q__place(self.position,self.vus,action,False) # 1 = HOUSE / 0 = OFFICE
        self.reward = reward
        self.draw_elements_on_canvas()
        #on reinitialise si il ny a plus assez despace vacan dans la ville
        if np.count_nonzero(self.map==0)<self.stop:
            self.position = self.select_next_cell()
            x0 = self.position[0]
            y0 = self.position[1]
            OBSMAP = self.getMap(self.map,x0,y0)
            self.vus=OBSMAP
            #on sauvegarde le nombre de chaque batiment predit pour avoir une ider de la diversiter predite
            conteurDeDiversiter=[]
            conteurDeDiversiter.append(len(self.offices))
            conteurDeDiversiter.append(len(self.houses))
            conteurDeDiversiter.append(len(self.parks))
            conteurDeDiversiter.append(len(self.coms))
            with open(str(self.path)+str('DivDqn.txt'), "a+") as fhandle:
                fhandle.write(str(conteurDeDiversiter)+'\n')
            #on sauvegarde la reward totale sur toute la ville, avec comme inpute la ville dans sont etat final(finalement pas)
            #self.GetValue()
            return OBSMAP, reward, True, {}

        self.position = self.select_next_cell()
        x0=self.position[0]
        y0=self.position[1]
        OBSMAP=self.getMap(self.map,x0,y0)
        self.vus=OBSMAP
        return OBSMAP, reward, False, {}
        
    def __draw_element_on_canvas(self, y, x, color):
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)

        # fit element to the canvas
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            for i in range(x * drawing_width, x * drawing_width + drawing_width):
                try : self.canvas[i, j] = color
                except IndexError : 
                    print('error')
                    pass
                
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            try : self.canvas[x * drawing_width, j] = GREY
            except IndexError : pass
            
            
        for i in range(x * drawing_width, x * drawing_width + drawing_width):
            try : self.canvas[i, y * drawing_height] = GREY
            except IndexError : pass
        pass

    def __draw_area_position(self, thickness = 3): # thickness must be odd 
        SIZE = self.observation_shape[0] // 2
        
        y, x = self.position
        thickness_range = range(- (thickness // 2), thickness // 2 + 1)
        
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)
        
        for j in range((y - SIZE) * drawing_height, (y + SIZE) * drawing_height + drawing_height):
            try :
                for t in thickness_range:
                    self.canvas[(x - SIZE) * drawing_width + t, j] = BLACK
                    self.canvas[(x + SIZE + 1) * drawing_width + t, j] = BLACK
            except IndexError : pass

        for i in range((x - SIZE) * drawing_width, (x + SIZE) * drawing_width + drawing_width):
            try :
                for t in thickness_range:
                    self.canvas[i, (y - SIZE) * drawing_height + t] = BLACK
                    self.canvas[i, (y + SIZE + 1) * drawing_height + t] = BLACK
            except IndexError : pass
            
        pass
    
    def __draw_player_position(self, thickness = 3): # thickness must be odd 
        y, x = self.position
        thickness_range = range(- (thickness // 2), thickness // 2 + 1)
        
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)
        
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            try :
                for t in thickness_range:
                    self.canvas[x * drawing_width + t, j] = BLACK
                    self.canvas[(x + 1) * drawing_width + t, j] = BLACK
            except IndexError : pass

        for i in range(x * drawing_width, x * drawing_width + drawing_width):
            try :
                for t in thickness_range:
                    self.canvas[i, y * drawing_height + t] = BLACK
                    self.canvas[i, (y + 1) * drawing_height + t] = BLACK
            except IndexError : pass
            
        pass
    
    def draw_elements_on_canvas(self):
        
        # draw each element of the map
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                
                color = WHITE
                if   self.map[y, x] == OFFICE : color = BLUE
                elif self.map[y, x] == HOUSE  : color = RED
                elif self.map[y, x] == PARK  : 
                    color = GREEN
                elif self.map[y, x] == COM  : 
                    color = [255,255,0]
                
                self.__draw_element_on_canvas(y, x, color)
            pass
               
        # draw player's position
        self.__draw_player_position()
        self.__draw_area_position()
        pass
    
    def render(self, mode = "console"):
        if mode == "human" :
            cv2.putText(self.canvas, str(self.reward), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("", self.canvas)
            cv2.waitKey(1)
            return self.canvas# uncomante si pr la video
        if mode == "console" :
            print(self.position)
    
    def close(self):
        pass
    
if __name__ == "__main__":
    env = City((20, 20),path='./')