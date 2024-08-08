"""
code: ebamareaprint.py
created by: William Frieden Templeton 
email: wfrieden@andrew.cmu.edu, williamfriedentempleton@gmail.com
org: Carnegie Mellon University
date: March, 2024


Hard code spots: 
ln 149 - Hard coded serial number extractor for the csv files
ln 173 - Hard coded filename lookup

Inputs for the file path in main

"""


# This just takes the x,y positions and does timed spots like a normal spot melting file
def PlanOptiSpots(x, y, power, dwell, number, loc):
    # from square import square
    from obplib.Line import Line
    from obplib.Point import Point
    from obplib import FileHandler
    from obplib.TimedPoints import TimedPoints
    from obplib.Beamparameters import Beamparameters

    spot_size = 94
    # time = int(time * 1e6)
    points = []
    for i in range(len(x)):
        points.append(Point(
            x[i],
            y[i]))
    print(np.sum(dwell[:])*1e3)
    dwells = [int(x*1e9) for x in dwell]
    print(np.sum(dwells)*1e-6)

    # Generate timed points OBP object and write to file
    spots = TimedPoints(points, dwells, Beamparameters(spot_size, power))
    pattern = [spots]
    FileHandler.write_obp(
        pattern,
        f"{number}_{loc}.obp"
    )

# This takes in x, y spots and adds line segments to ensure short dwell times are not truncated
# This was corrected in the optimizer, so the travel time from one spot to the next should be included in the dwell time (i.e., use PlanOptiSpots)
def PlanOptiLineSpots(x, y, power, dwell, scanname):
    # from square import square
    from obplib.Line import Line
    from obplib.Point import Point
    from obplib import FileHandler
    from obplib.TimedPoints import TimedPoints
    from obplib.Beamparameters import Beamparameters

    spot_size = 94
    # time = int(time * 1e6)
    # times = []
    # points = []
    backspotx = 0
    backspoty = 0
    write_to_obpj(backspotx*1e-6, backspoty*1e-6, x[0], y[0], 'line', spot_size, 3000, 4e9, scanname, islast=False)
    for i in range(len(x)-2):
        write_to_obpj(x[i], y[i], None, None, 'timedpoints', spot_size, 3000, 1000, scanname, islast=False)
        write_to_obpj(x[i],y[i],x[i+1],y[i+1],'line',spot_size, 3000, 1e9,scanname, islast=False)
    write_to_obpj(x[-2], y[-2], None, None, 'timedpoints', spot_size, 3000, 1000, scanname, islast=False)
    write_to_obpj(x[-2],y[-2],x[-1],y[-1],'line',spot_size, 3000, 1e9,scanname, islast=False)
    write_to_obpj(x[-1], y[-1], None, None, 'timedpoints', spot_size, 3000, 1000, scanname, islast=True)


# Just a writer.
def write_to_obpj(x1, y1, x2, y2, scantype, spot_size, power, timing, scanname, islast):
    if scantype == 'line':
        with open(f"{scanname}.obpj", 'a') as scanfile:
            scanfile.write("{\n")
            scanfile.write('    "line": {\n')
            scanfile.write('        "params": {\n')
            scanfile.write(f'            "spotSize": {float(spot_size)},\n')
            scanfile.write(f'            "beamPower": {float(power)}\n')
            scanfile.write('        },\n')
            scanfile.write(f'        "x0": {float(x1*1e6):.01f},\n')
            scanfile.write(f'        "y0": {float(y1*1e6):.01f},\n')
            scanfile.write(f'        "x1": {float(x2*1e6):.01f},\n')
            scanfile.write(f'        "y1": {float(y2*1e6):.01f},\n')
            scanfile.write(f'        "speed": "{int(timing)}"\n')
            scanfile.write('    }\n')
            scanfile.write('},\n')
            scanfile.close()
    if scantype == 'timedpoints':
        with open(f"{scanname}.obpj", 'a') as scanfile:
            scanfile.write("{\n")
            scanfile.write('    "timedpoints": {\n')
            scanfile.write('        "params": {\n')
            scanfile.write(f'            "spotSize": {float(spot_size)},\n')
            scanfile.write(f'            "beamPower": {float(power)}\n')
            scanfile.write('        },\n')
            scanfile.write('        "points": [\n')
            scanfile.write('            {\n')
            scanfile.write(f'                "x": {float(x1*1e6):.01f},\n')
            scanfile.write(f'                "y": {float(y1*1e6):.01f},\n')
            scanfile.write(f'                "t": {int(timing)}\n')
            scanfile.write('            }\n')
            scanfile.write('        ]\n')
            scanfile.write('    }\n')
            if islast==True:
                scanfile.write('}\n')
            else:
                scanfile.write('},\n')
            scanfile.close()


# Function to adjust the positions of patches - just spaces out things
def adjust_patch_positions(plan, patch_width=5e3, patch_height=5e3, buffer_space=2e3, grid_rows=6, grid_cols=5):
    adjusted_data = {}
    row_index = 0
    col_index = 0
    # counter = 0

    for patch in plan:
        # Calculate the grid position
        row = row_index // grid_rows
        col = col_index 

        # Calculate the shift in mm, then convert to the same units as in the CSV (looks like meters)
        x_shift = (col * (patch_width + buffer_space))   # Converting mm to meters
        y_shift = (row * (patch_height + buffer_space)) 

        # Apply the shift
        plan[patch]['X'] = plan[patch]['X']*1e6# x_shift  - 20e3
        plan[patch]['Y'] = plan[patch]['Y']*1e6# y_shift  - 15e3

        adjusted_data[patch] = plan[patch]

        row_index += 1
        col_index += 1
        if col_index > grid_cols:
            col_index = 0
        # counter+=1
        # print(counter)

    return adjusted_data

def plotplan(x,y):
    fig = plt.figure()
    plt.scatter(x, y)
    plt.show()

def extract_first_number(filename):
    match = re.search(r'scan_strat_([\d\.]+)_', filename)
    if match:
        return float(match.group(1))
    return 0  # Default value if no number is found

def sorted_files(filenames):
    filenames.sort(key=extract_first_number)
    return filenames

def stack_patch_data(plan):
    # Collect and concatenate all x, y, z values from each patch
    x_all = np.concatenate([patch['X'] for patch in plan.values()])
    y_all = np.concatenate([patch['Y'] for patch in plan.values()])
    z_all = np.concatenate([patch['T'] for patch in plan.values()])

    return x_all, y_all, z_all

if __name__ == "__main__":
    import numpy as np
    import obplib

    optimal_spots = {}
    # Machine Parameters
    # filename = 'scan_strat_1.0_1.0.csv'
    filename = 'scan_strat_optimized.csv'

    # This loads the csv files into a dictionary
    # It's a bit bulky, but was originally set up to load multiple files. 
    plan = {}
    plan[filename] = np.loadtxt(filename, delimiter=',',skiprows=1)
    plan[filename] = {'X': plan[filename][:, 0], 'Y': plan[filename][:, 1], 'T': plan[filename][:, 2]}

    # Breaks everything up into spots, records the min/max location of the obj (it's set up for constant area parts in the part spacing operation)
    xspots, yspots, _ = stack_patch_data(plan)
    xmin, xmax, ymin, ymax = np.min(xspots), np.max(xspots), np.min(yspots), np.max(yspots)
    print(f"{xmin}, {xmax}, {ymin}, {ymax}")
    
    # Spacing of the parts in the layer, again it's set up for multiple parts. 
    spacing = 0.0075*1e6
    xwidth = (xmax-xmin)*1e6
    ywidth = (ymax-ymin)*1e6
    print(f"{xwidth*1e-3}, {ywidth*1e-3}")
    centery = 1e6*(ymax-ymin+0.0025)/2
    centerx = 1e6*(xmax-xmin+0.0025)/2
    dwell = plan[filename]['T'][0]
    totaltime = plan[filename]['T'].sum()
    print(plan[filename]['T'])
    print(f"Runtime: {totaltime*1e3}ms")

    # This is for the 6x6 grid we did, edit position as needed sorry for the hard code :D 
    xloc, yloc = 0,0
    positions = {
        (0,0): [-xwidth-spacing, -ywidth-spacing],
        (0,1): [-xwidth-spacing, 0],
        (0,2): [-xwidth-spacing, +ywidth+spacing],
        (0,3): [-xwidth-spacing, 2*(+ywidth+spacing)],
        (1,0): [0, -ywidth-spacing],
        (1,1): [0, 0],
        (1,2): [0, +ywidth+spacing],
        (1,3): [0, 2*(+ywidth+spacing)],
        (2,0): [+xwidth+spacing, -ywidth-spacing],
        (2,1): [+xwidth+spacing, 0],
        (2,2): [+xwidth+spacing, +ywidth+spacing],
        (2,3): [+xwidth+spacing, 2*(+ywidth+spacing)]
    }

    PlanOptiSpots(
        xspots*1e6-centerx+positions[xloc,yloc][0],
        yspots*1e6-centery+positions[xloc,yloc][1],
        power=3000,
        dwell=plan[filename]['T'],
        number=filename.split('.')[0], # Both number and loc show up in the file name 
        loc = f"{xloc}_{yloc}"
        )

# The commented out stuff was from when we were doing spot and line kind of stuff. 

    # scanname = filename.split('.')[0]
    # scanfile = open(f'{scanname}.obpj', 'w')
    # scanfile.write('{\n')
    # scanfile.write('"OBP":[\n')
    # scanfile.close()
    # print(xspots)
    # PlanOptiLineSpots(xspots+15e-3, yspots-2.5e-3, 3000, dwell, scanname)
    # scanfile = open(f'{scanname}.obpj', 'a')
    # scanfile.write(']\n')
    # scanfile.write('}')

# if __name__ == "__main__":
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import re
#     laser_power = 1000
#     # directory = "path/to/csv/files"
#     patches_per_row = 5  # since 50mm/5mm = 10
#     patches = []
#     plan = {}

#     filanems = []
#     filenames = os.listdir()
#     filenames = [file for file in filenames if file.endswith(".csv")]
#     filenames = sorted_files(filenames)
#     for filename in filenames:
#         if filename.endswith(".csv"):
#             plan[filename] = np.loadtxt(filename, delimiter=',', skiprows=1)
#             plan[filename] = {'X': plan[filename][:, 0], 'Y': plan[filename][:, 1], 'T': plan[filename][:, 2]}
#     plan = adjust_patch_positions(plan)
#     x_spots, y_spots, _ = stack_patch_data(plan)
#     plotplan(x_spots, y_spots)
#     PlanOptiSpots(x_spots, y_spots, 3000, 20, '30by30')