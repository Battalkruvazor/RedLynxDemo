

def parse_grid(grid_str):
    grid = [[(ord(s) - ord('a') if s != '#' else 4) for s in r] for r in grid_str.split()]
    return grid

def one_hot(cell):
    return [1 if cell == i else 0 for i in range(5)]

def rotate_cell(cell, rot = 1):
    if cell<4:
        return (cell+rot)%4
    else: 
        return cell


