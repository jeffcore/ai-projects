from enum import Enum
import random

class Prize(Enum):
    LOSING = 0
    WINNING = 1

class Door:
    def __init__(self):
        self.is_open = False
        self.prize = Prize.LOSING

    def set_open(self, is_open):
        self.is_open = is_open
    
    def set_prize(self, prize):
        self.prize = prize
    def __repr__(self):  
        return 'Door Open ' + str(self.is_open) + ' ' + str(self.prize)

    def __str__(self):
        return 'Door Open ' + self.is_open + ' prize ' + self.prize

class MontyHall:
    def __init__(self):
        self.rounds = 0
        self.results_wins = 0        
        self.doors = ['', Door(), Door(), Door()]
        self.doors_contents  = [0, Prize.LOSING, Prize.LOSING, Prize.LOSING]
        self.is_winner = True

    def run(self):
        self.print_welcome()
        self.set_up_doors()        
        self.print_door()

        # select first door
        selected_door = self.get_door_number()
        print(f'You choose door {selected_door}. I opened another door.\n')
        # open other door
        self.open_other_door(selected_door)           
       
        self.print_door()
        
        selected_door = self.get_door_number()
        self.open_all_doors()
        self.print_door()
        if self.check_winner(selected_door):
            print('You won')                        
        else:
            print('You lost')    
            self.is_winner = False
        

    def print_welcome(self):
        print('''Welcome to the Monty Hall Problem!\n     
        You get to select a door. Then I will open another door. 
        Then you can choose to change the door you selected.       
        ''')

    def print_door(self):
        print('The Doors')
        doors = ''
        for i in range(1,4):
            if not self.doors[i].is_open:
                doors += '|' + str(i) + '| '
            else:
                if self.doors[i].prize == Prize.WINNING:
                    doors += '|Car| '
                else:
                    doors += '|$10| '
        print(doors , '\n')            

    def get_door_number(self):
        while True:
            result = input("Pick A Door \n >>> ")
            print('\n')
            if result.isdigit() and int(result) >= 1  and int(result) <= 3:
                return int(result)                
            else:
                print('Please pick one of the following number (1,2,3)')

    def set_up_doors(self):
        result = random.randrange(1,4)
        self.doors[result].set_prize(Prize.WINNING)        
    
    def check_winner(self, door_selected):
        if self.doors[door_selected].prize == Prize.WINNING:
            return True
        else:
            return False

    def open_other_door(self, door_selected):    
        while True:
            door_number = random.randrange(1,4)
            if door_number != door_selected and not self.doors[door_number].is_open and self.doors[door_number].prize == Prize.LOSING:
                self.doors[door_number].set_open(True)
                break

    def open_all_doors(self):
        for i in range(1, 4):
            self.doors[i].is_open = True


def main():
    number_games = 0
    number_wins = 0

    while True:
        monty = MontyHall()
        monty.run()
        number_games += 1
        if monty.is_winner:
            number_wins += 1

        print(f'\nGames Played: {number_games}\nWins: {number_wins}\nWin Percentage: {(number_wins/number_games)*100:.2f}%\n')

        result = input('do you want to play again? >>> (y or n) ')
        if result ==  'n' or result == 'no':
            break

if __name__ == '__main__':
    main()
