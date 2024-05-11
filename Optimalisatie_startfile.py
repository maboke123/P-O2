import pygame
import sys
import time

ACCEPT = 'accept'
ABANDON = 'abandon'
CONTINUE = 'continue'

#Kleuren bepalen
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255,255,0)

#Constanten bepalen
BLOCK_SIZE = 50
MARGIN = 1

def initiate_board(nbRows,nbCols):
    matrix = []
    if nbRows < 1:
        return None
    if nbCols < 1:
        return None
    for i in range(nbRows):
        list = []
        for i in range(nbCols):
            space = ' '
            list.append(space)
        matrix.append(list)
    return matrix
def putGreen(board,x,y):
    nbRows = len(board)
    nbCols = len(board[1])
    green = "G"
    board[x][y] = green
    if x > nbRows:
        return 1
    if y > nbCols:
        return 1
def putRed(board, x, y):
    nbRows = len(board)
    nbCols = len(board[1])
    red = "R"
    board[x][y] = red
    if x > nbRows:
        return 1
    if y > nbCols:
        return 1
def getGreens(board):
    green_positions = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 'G':
                green_positions.append((i,j))
    return green_positions
def getReds(board):
    red_positions = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 'R':
                red_positions.append((i,j))
    return red_positions
def getLegalNeighbours(board,pos):
    # Grenzen van het bord bepalen
    max_rows = len(board)
    max_colums = len(board[1])

    # Zorgen dat pos altijd een tuple is
    if isinstance(pos, list) and len(pos) == 1 and isinstance(pos[0], tuple):
        pos = pos[0]

    if pos == 0:
        return 1

    legal_neighbours = []

    if isinstance(pos,tuple):
        neighbours_x = [pos[0] - 1, pos[0], pos[0] + 1]
        neighbours_y = [pos[1] - 1, pos[1], pos[1] + 1]

        # loopen over alle posities rond start positie
        for i in neighbours_x:
            for j in neighbours_y:

                # Eigen positie niet meetellen
                if i == pos[0] and j == pos[1]:
                    legal_neighbours = legal_neighbours

                # Checken of waarde buiten de grenzen van het bord liggen
                elif i < 0 or i >= max_rows:
                    legal_neighbours = legal_neighbours

                elif j < 0 or j >= max_colums:
                    legal_neighbours = legal_neighbours

                # Checken of de buur een rode pin is
                elif board[i][j] == "R":
                    legal_neighbours = legal_neighbours

                # Diagonale posities er uit halen
                elif (i == pos[0] - 1 and j == pos[1] - 1) or (i == pos[0] + 1 and j == pos[1] + 1) or (i == pos[0] - 1 and j == pos[1] + 1) or (i == pos[0] + 1 and j == pos[1] - 1):
                    legal_neighbours = legal_neighbours

                # Koppel van geldige buur samenzetten per koppel en in een lijst
                else:
                    legal_neighbours.append((i,j))
    return legal_neighbours
def calculateTime(route):
    TIMESTRAIGHT = 2.61857
    TIMETURN = 3.56333
    total_time = 0
    total_turns = 0
    total_straights = 0
    for i in range(len(route) - 1):
        # Rechtdoor: als x gelijk blijft en y met 1 verandert
        if route[i][0] == route[i + 1][0] and route[i][1] == route[i + 1][1] + 1:
            total_time += TIMESTRAIGHT
            total_straights += 1
        if route[i][0] == route[i + 1][0] and route[i][1] == route[i + 1][1] - 1:
            total_time += TIMESTRAIGHT
            total_straights += 1

        # Rechtdoor: als y gelijk blijft en x met 1 verandert
        if route[i][0] == route[i + 1][0] - 1 and route[i][1] == route[i + 1][1]:
            total_time += TIMESTRAIGHT
            total_straights += 1
        if route[i][0] == route[i + 1][0] + 1 and route[i][1] == route[i + 1][1]:
            total_time += TIMESTRAIGHT
            total_straights += 1

    for i in range(len(route) - 2):
        # Draai: als eerst x gelijk blijft en dan y gelijk blijft
        if (route[i][0] == route[i + 1][0] and route[i][1] != route[i + 1][1]) and (route[i + 1][0] != route[i + 2][0] and route[i + 1][1] == route[i + 2][1]):
            total_time += TIMETURN
            total_turns += 1

        # Draai: als eerst y gelijk blijft en dan x gelijk blijft
        if (route[i][0] != route[i + 1][0] and route[i][1] == route[i + 1][1]) and (route[i + 1][0] == route[i + 2][0] and route[i + 1][1] != route[i + 2][1]):
            total_time += TIMETURN
            total_turns += 1

        # 180 draai: als i == i + 2
        if route[i][0] == route[i + 2][0] and route[i][1] == route[i + 2][1]:
            total_time += 2 *  TIMETURN
            total_turns += 2

    return total_time
def makedistancematrix(start, finish, board):
    distancematrix = []
    temp = []

    green_positions = getGreens(board)
    # Alle afstanden voor start berekenen
    # Afstand tussen start en zichzelf
    temp.append(0)

    # Afstand berekenen tussen start en alle groene posities
    for i in range(len(green_positions)):
        distance = calculateTime(dfs_recursive(green_positions[i], start, [], None, board))
        temp.append(distance)

    # Afstand berekenen tussen start en finish
    distance = calculateTime(dfs_recursive(start, finish, [], None, board))
    temp.append(distance)

    distancematrix.append(temp)

    # Alle afstanden voor groene posities berekenen
    for i in range(len(green_positions)):
        temp = []
        # afstand tussen groene posities en start
        distance = calculateTime(dfs_recursive(green_positions[i], start, [], None, board))
        temp.append(distance)

        # afstand tussen groene posities onderling
        for j in range(len(green_positions)):
            distance = calculateTime(dfs_recursive(green_positions[i], green_positions[j], [], None, board))
            temp.append(distance)

        # afstand tussen groene posities en finish
        distance = calculateTime(dfs_recursive(green_positions[i], finish, [], None, board))
        temp.append(distance)

        distancematrix.append(temp)

    # Alle afstanden voor finish berekenen
    temp = []

    # Afstand tussen start en finish
    distance = calculateTime(dfs_recursive(start, finish, [], None, board))
    temp.append(distance)

    # Afstand berekenen tussen start en alle groene posities
    for i in range(len(green_positions)):
        distance = calculateTime(dfs_recursive(green_positions[i], finish, [], None, board))
        temp.append(distance)

    # Afstand berekenen tussen finish en zichzelf
    temp.append(0)

    distancematrix.append(temp)
    return (distancematrix)
def opstart_costfunctie():
    green_positions = getGreens((board))

    # Alle nodes een cijfer geven om makkelijker in de distancematrix te indexen
    keys = []
    keys.append(start)

    for i in green_positions:
        keys.append(i)

    if not start == finish:
        keys.append(finish)

    values = []
    temp = 0

    for j in range(len(green_positions) + 2):
        values.append(temp)
        temp += 1

    cijfers = {keys[i]:values[i] for i in range(len(keys))}
    return cijfers
def tuple_to_cijfers(to_visit):
    return [cijfers[item] for item in to_visit]
def cijfers_to_tuples(to_visit):
    return [cijfers_reversed[item] for item in to_visit]
def makeInstructionfile(route, board):
    file = open('instructions.txt','w')
    for pos in route:
        pick = "False"
        if board[pos[0]][pos[1]] == "G":
            pick = "True"
        outputstring = str(pos[0])+", "+str(pos[1]) +", " + pick + "\n"
        file.write(outputstring)
    file.close()
def fastestRoute(start, board, finish):
    #Kortste route vinden tusse alle groene posities en dan van laatste groene positie naar einde
    first_part = shortest_route(start, board, []) #+ dfs_recursive((0,0), finish, [], None, board)
    second_part = dfs_recursive(first_part[-1], finish, [], None, board)
    total = first_part + second_part

    total_shortest_route_no_duplicates = [start]

    for i in range(len(total)):
        if total[i] != total[i - 1]:
            total_shortest_route_no_duplicates.append(total[i])

    return total_shortest_route_no_duplicates
def shortest_route(current, board, total_shortest_route):
    #kortste route vinden tussen alle groene posities
    shortest_path = []
    index_shortest_path = None
    for i in green_positions:
        path = dfs_recursive(current, i, [], None, board)
        if path and (len(shortest_path) == 0 or calculateTime(path) < calculateTime(shortest_path)):
            shortest_path = path
            index_shortest_path = i

    total_shortest_route += shortest_path

    current = index_shortest_path

    if index_shortest_path in green_positions:
        green_positions.remove(index_shortest_path)

    if len(green_positions) > 0:
        shortest_route(current, board, total_shortest_route)

    total_shortest_route_no_duplicates = []

    for i in range(len(total_shortest_route)):
        if total_shortest_route[i] != total_shortest_route[i - 1]:
            total_shortest_route_no_duplicates.append(total_shortest_route[i])

    return total_shortest_route_no_duplicates
def dfs_recursive(current, finish, visited, shortest_path, board):
    #Kortste positie tussen 2 punten vinden
    if current == finish:
        return visited + [current]

    visited.append(current)

    neighbors = getLegalNeighbours(board, current)

    shortest_path = None

    for neighbor in neighbors:
        if neighbor not in visited:
            path = dfs_recursive(neighbor, finish, visited[:], shortest_path, board)
            if path and (shortest_path is None or calculateTime(path) < calculateTime(shortest_path)):
                shortest_path = path

    return shortest_path
def visualize_maze(board,route):
    pygame.init()

    #Scherm opzetten
    width = len(board[0]) * BLOCK_SIZE + (len(board[0]) - 1) * MARGIN
    height = len(board) * BLOCK_SIZE + (len(board) - 1) * MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Board")

    #Hoofdloop opstellen met tijd
    done = False
    clock = pygame.time.Clock()

    index = 0
    color_index = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True


        screen.fill(BLACK)

        #Bord opstellen
        for y in range(len(board)):
            for x in range(len(board[0])):
                if board[y][x] == " ":
                    color = WHITE
                elif board[y][x] == "R":
                    color = RED
                elif board[y][x] == "G":
                    color = GREEN
                rect = pygame.Rect(x * (BLOCK_SIZE + MARGIN), y * (BLOCK_SIZE + MARGIN), BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(screen, color, rect)

        #Route afgaan
        if index < len(route):
            y, x = route[index]
            rect = pygame.Rect(x * (BLOCK_SIZE + MARGIN), y * (BLOCK_SIZE + MARGIN), BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, YELLOW, rect)


        pygame.display.flip()
        clock.tick(3)


        index += 1

        if index >= len(route):
            #time.sleep(10)
            done = True

    pygame.quit()
def examine(partial_solution, distancematrix):
    #Checken of dezelfde node niet nog is bezocht wordt
    for i in range(len(partial_solution)):
        for j in range(i + 1,len(partial_solution)):
            if partial_solution[i] == partial_solution[j]:
                return ABANDON

    #Checken of de oplossing alle nodes bevat
    if len(partial_solution) == len(distancematrix):
        #Checke of de laatste node wel degelijk de finish node is
        if partial_solution[-1] == len(distancematrix) - 1:
            return ACCEPT
        return ABANDON

    return CONTINUE
def extend(partial_solution, distancematrix):
    new_partial_solutions = []

    for i in range(len(distancematrix)):
        new_partial_solution = list(partial_solution.copy())
        new_partial_solution.append(i)
        new_partial_solutions.append(new_partial_solution)

    return new_partial_solutions
def solve(partial_solution, distancematrix):
    stack = [partial_solution]

    shortest_distance = float('inf')
    shortest_path = []

    while stack:
        current_partial_solution = stack.pop()

        exam = examine(current_partial_solution, distancematrix)

        if exam == ACCEPT:
            #Afstand van een mogelijke route bepalen
            current_distance = 0
            for i in range(len(current_partial_solution) - 1):
                current_distance += distancematrix[current_partial_solution[i]][current_partial_solution[i + 1]]

            #Korste afstand en route bepalen
            if current_distance < shortest_distance:
                shortest_distance = current_distance
                shortest_path = current_partial_solution

        elif exam == CONTINUE:
            for p in extend(current_partial_solution, distancematrix):
                stack.append(p)

    return(shortest_path)
def examine2(partial_solution, size_of_problem):
    #Checken of dezelfde node niet nog is bezocht wordt
    for i in range(len(partial_solution)):
        for j in range(i + 1,len(partial_solution)):
            if partial_solution[i] == partial_solution[j]:
                return ABANDON

    #Checken of de oplossing alle nodes bevat
    if len(partial_solution) == size_of_problem:
        #Checke of de laatste node wel degelijk de finish node is
        if partial_solution[-1] == size_of_problem - 1:
            return ACCEPT
        return ABANDON

    return CONTINUE
def extend2(partial_solution, size_of_problem):
    new_partial_solutions = []

    for i in range(size_of_problem):
        new_partial_solution = list(partial_solution.copy())
        new_partial_solution.append(i)
        new_partial_solutions.append(new_partial_solution)

    return new_partial_solutions
def solve2(partial_solution, size_of_problem):
    stack = [partial_solution]

    shortest_distance = float('inf')
    shortest_path = []

    while stack:
        current_partial_solution = stack.pop()

        exam = examine2(current_partial_solution, size_of_problem)

        if exam == ACCEPT:

            # Kortste route omzetten van getallen naar de coördinaten
            order_to_visit_coords = []
            for i in range(len(current_partial_solution) - 1):
                order_to_visit_coords.append(cijfers_reversed[current_partial_solution[i]])
            order_to_visit_coords.append(finish)

            # Totale korste route terug bepalen tussen de punten
            correct_route = []
            for i in range(len(order_to_visit_coords) - 1):
                correct_route += (
                    dfs_recursive(order_to_visit_coords[i], order_to_visit_coords[i + 1], [], None, board))

            # Dubbele coördinaten uit de route halen
            correcte_route_zonder_dupes = []
            for i in range(len(correct_route)):
                if correct_route[i] != correct_route[i - 1]:
                    correcte_route_zonder_dupes.append(correct_route[i])

            #Lengte van de huidige route bepalen
            current_distance = calculateTime(correcte_route_zonder_dupes)

            #Korste afstand en route bepalen
            if current_distance < shortest_distance:
                shortest_distance = current_distance
                shortest_path = correcte_route_zonder_dupes

        elif exam == CONTINUE:
            for p in extend2(current_partial_solution, size_of_problem):
                stack.append(p)

    print(shortest_path)
    print('Time:', shortest_distance)
    return(shortest_path)


#Bord opzetten en groene en rode schijven plaatsen
board = initiate_board(4, 6)
putGreen(board, 1, 2)
putGreen(board, 2, 0)
putGreen(board, 3, 4)
putGreen(board,0, 5)
putGreen(board,0, 3)
putRed(board, 0, 4)
putRed(board, 1, 1)
putRed(board, 3, 2)
putRed(board, 3, 5)
green_positions = getGreens(board)

#Bord printen
print()
print('Bord opstelling')
for i in range(len(board)):
    print(board[i])
print()

#Start en finish bepalen
start = (0, 0)
finish = (0, 0)

#####################################
#Approach oplossing
route = fastestRoute(start,board,finish)
print('Route (approach)')
print(route)
print('Time:',calculateTime(route))

#Visualisren oplossing
visualize_maze(board,route)


#####################################
#Volledig juiste oplossing (zonder rekening te houden met draaien op groene schijven)

#Distancematrix opstellen
distancematrix = makedistancematrix(start, finish, board)

#Distancematrix printen
print()
print('Distancematrix')
for i in distancematrix:
    print(i)
print()

#Dicts opstellen om van coördinaten naar getallen te gaan en omgekeerd
#Van Coördinaten naar getallen:
cijfers = opstart_costfunctie()
#Van getallen naar Coördinaten:
cijfers_reversed = {v: k for k, v in cijfers.items()}

#Via Distancematrix kortste route bepalen
partial_solution = [0]
shortest_path = solve(partial_solution, distancematrix)


#Kortste route omzetten van getallen naar de coördinaten
order_to_visit_coords = []
for i in range(len(shortest_path) - 1):
    order_to_visit_coords.append(cijfers_reversed[shortest_path[i]])
order_to_visit_coords.append(finish)


#Totale korste route terug bepalen tussen de punten
correct_route = []
for i in range(len(order_to_visit_coords) - 1):
    correct_route += (dfs_recursive(order_to_visit_coords[i],order_to_visit_coords[i + 1],[],None, board))

#Dubbele coördinaten uit de route halen
correcte_route_zonder_dupes = []
for i in range(len(correct_route)):
    if correct_route[i] != correct_route[i - 1]:
         correcte_route_zonder_dupes.append(correct_route[i])

print('Route (correct version, not an approach)')
print(correcte_route_zonder_dupes)
print('Time:', calculateTime(correcte_route_zonder_dupes))
print()

#Visualisren oplossing
visualize_maze(board,correcte_route_zonder_dupes)

#####################################
#Volledig juiste oplossing
partial_solution = [0]
green_positions = getGreens(board)
size_of_problem = len(green_positions) + 2
print('Route (correct version with turning on green positions, not an approach)')
correct_route_with_turning_on_green_positions = solve2(partial_solution, size_of_problem)

visualize_maze(board,correct_route_with_turning_on_green_positions)

