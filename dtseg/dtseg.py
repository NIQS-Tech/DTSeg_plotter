import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib.path import Path
from shapely.geometry import Point, MultiPolygon, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_DTSEG(input_data: pd.DataFrame = None, plot_size: int=600, point_size: int = 20, Show_zone_labels: bool = True, save_graph: bool= True, graph_title: str = None, highlight_zone: str = None):


    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax = plt.gca()

    ################# Zone A-B ##################################
    # Vertical line from (62.5, 0) to (62.5, 50)
    plt.plot([62.5, 62.5], [0, 50], 'k-', linewidth=2)

    # Horizontal line from (0, 60) to (50, 60)
    plt.plot([0, 50], [60, 60], 'k-', linewidth=2)

    # Dashed line y = x
    x_vals = np.linspace(0, 600, 200)
    plt.plot(x_vals, x_vals, 'k--', linewidth=1.5, label='y = x')

    # Black line y = 0.8x starting from (62.5, 50)
    x1 = np.linspace(62.5, 600, 100)
    y1 = 0.8 * x1 - 0.8 * 62.5 + 50  # Shift so it starts at (62.5, 50)
    plt.plot(x1, y1, 'k-', linewidth=2)

    # Black line y = 1.2x starting from (50, 60)
    x2 = np.linspace(50, 600, 100)
    y2 = 1.2 * x2 - 1.2 * 50 + 60  # Shift so it starts at (50, 60)
    plt.plot(x2, y2, 'k-', linewidth=2)

    ################# Zone B-C ##################################
    # Vertical line from (97.5, 0) to (97.5, 50)
    plt.plot([97.5, 97.5], [0, 50], 'k-', linewidth=2)

    # Horizontal line from (0, 86.5) to (50, 86.5)
    plt.plot([0, 50], [86.5, 86.5], 'k-', linewidth=2)


    # Black line y = 514/1005x starting from (97.5, 50)
    x3 = np.linspace(97.5, 600, 100)
    y3 = 514/1005 * x3 - 514/1005 * 97.5 + 50  # Shift so it starts at (97.5, 50)
    plt.plot(x3, y3, 'k-', linewidth=2)

    # Black line y = 1027/594x starting from (50, 86.5)
    x4 = np.linspace(50, 600, 100)
    y4 = 1027/594 * x4 - 1027/594 * 50 + 86.5  # Shift so it starts at (50, 86.5)
    plt.plot(x4, y4, 'k-', linewidth=2)


    ################# Zone C-D ##################################
    # Vertical line from (153, 0) to (153, 50)
    plt.plot([153, 153], [0, 50], 'k-', linewidth=2)

    # Horizontal line from (0, 124) to (50, 124)
    plt.plot([0, 50], [124, 124], 'k-', linewidth=2)


    # Black line y = 49/149x starting from (153, 50)
    x5 = np.linspace(153, 600, 100)
    y5 = 49/149 * x5 - 49/149 * 153 + 50  # Shift so it starts at (153, 50)
    plt.plot(x5, y5, 'k-', linewidth=2)

    # Black line y = 476/191x starting from (50, 124)
    x6 = np.linspace(50, 600, 100)
    y6 = 476/191 * x6 - 476/191 * 50 + 124  # Shift so it starts at (50, 124)
    plt.plot(x6, y6, 'k-', linewidth=2)


    ################# Zone D-E ##################################
    # Vertical line from (238, 0) to (238, 50)
    plt.plot([238, 238], [0, 50], 'k-', linewidth=2)

    # Horizontal line from (0, 179) to (50, 179)
    plt.plot([0, 50], [179, 179], 'k-', linewidth=2)


    # Black line y = 38/181x starting from (238, 50)
    x7 = np.linspace(238, 600, 100)
    y7 = 38/181 * x7 - 38/181 * 238 + 50  # Shift so it starts at (238, 50)
    plt.plot(x7, y7, 'k-', linewidth=2)

    # Black line y = 421/117x starting from (50, 179)
    x8 = np.linspace(50, 600, 100)
    y8 = 421/117 * x8 - 421/117 * 50 + 179  # Shift so it starts at (50, 124)
    plt.plot(x8, y8, 'k-', linewidth=2)

    # --- FILL REGION: between vertical line at x=238 and y7, down to axes ---
    # Polygon points: (238,0) up to (238,50), then along y7 to (600, y7[-1]), then down to (600,0), then back to (238,0)
    x_fill = np.concatenate([[238], x7, [600], [600], [238]])
    y_fill = np.concatenate([[0], y7, [y7[-1]], [0], [0]])



    mask = x7 >= 153
    x7_sub = x7[mask]
    y7_sub = y7[mask]

    # Construct the polygon for the region to fill
    x_fill2 = np.concatenate([
        [153],           # Start at bottom left
        [153],           # Up to (153, 50)
        x5,              # Along the sloped line to (600, y5[-1])
        [600],           # Down to top of brown region at x=600
        x7_sub[::-1],    # Backwards along upper edge of brown region
        [238],
        [238],
    ])
    y_fill2 = np.concatenate([
        [0],             # Start at bottom left
        [50],            # Up to (153, 50)
        y5,              # Along the sloped line
        [y7_sub[-1]],    # Down to brown region at x=600
        y7_sub[::-1],    # Backwards along brown region
        [50],
        [0],
    ])

        

    x_fill3 = np.concatenate([
        [97.5],           # Start at bottom left
        [97.5],           # Up to (97.5, 50)
        x3,              # Along the sloped line to (600, y3[-1])
        [600],           # Down to top of brown region at x=600
        x5[::-1],    # Backwards along upper edge of red region
        [153],
        [153],
    ])
    y_fill3 = np.concatenate([
        [0],             # Start at bottom left
        [50],            # Up to (97.5, 50)
        y3,              # Along the sloped line
        [y5[-1]],    # Down to red region at x=600
        y5[::-1],    # Backwards along red region
        [50],
        [0],
    ])


    x_fill4 = np.concatenate([
        [62.5],           # Start at bottom left
        [62.5],           # Up to (62.5, 50)
        x1,              # Along the sloped line to (600, y1[-1])
        [600],           # Down to top of brown region at x=600
        x3[::-1],    # Backwards along upper edge of yellow region
        [97.5],
        [97.5],
    ])
    y_fill4 = np.concatenate([
        [0],             # Start at bottom left
        [50],            # Up to (62.5, 50)
        y1,              # Along the sloped line
        [y3[-1]],    # Down to red region at x=600
        y3[::-1],    # Backwards along rellow region
        [50],
        [0],
    ])

    x_fill5 = np.concatenate([
        [0],           
        [50],           
        x2,              
        [600],
        [600],               
        x1[::-1],    
        [62.5],
        [62.5],
        [0]
    ])
    y_fill5 = np.concatenate([
        [60],            
        [60],            
        y2,              
        [600],
        [y1[-1]], 
        y1[::-1],    
        [50],
        [0],
        [0]


    ])

    x_fill6 = np.concatenate([
        [0],           
        [50],           
        x4,      
        [x2[-1]],           
        x2[::-1],   
        [50],
        [0]
    ])
    y_fill6 = np.concatenate([
        [86.5],            
        [86.5],            
        y4,  
        [600],            
        y2[::-1],    
        [60],
        [60]
    ])

    x_fill7 = np.concatenate([
        [0],
        [50],          
        x6,     
        [x4[-1]],    
        x4[::-1],   
        [50],
        [0]
    ])
    y_fill7 = np.concatenate([
        [124],
        [124],
        y6,  
        [600],  
        y4[::-1],  
        [86.5],            
        [86.5],  
    ])

    x_fill8 = np.concatenate([

        [0],
        [50],
        x8,     
        [x6[-1]],   
        x6[::-1],   
        [50],
        [0],
    ])
    y_fill8 = np.concatenate([
        [179],
        [179],
        y8,  
        [600],  
        y6[::-1],  
        [124],
        [124],
    ])

    x_fill9 = np.concatenate([
        [0],
        [0],    
        [x8[-1]],  
        x8[::-1],   
        [50],
        [0],
    ])
    y_fill9 = np.concatenate([
        [179],
        [600], 
        [y8[-1]], 
        y8[::-1],  

        [179],
        [179],
    ])



    plt.fill(x_fill9, y_fill9, color='#D482A8', alpha=1, zorder=1)  
    plt.fill(x_fill8, y_fill8, color='#FF6A6A', alpha=1, zorder=1) 
    plt.fill(x_fill7, y_fill7, color='#FFCB6A', alpha=1, zorder=1)  
    plt.fill(x_fill6, y_fill6, color='#FDFD6D', alpha=1, zorder=1)  
    plt.fill(x_fill5, y_fill5, color='#6ABC6A', alpha=1, zorder=1)  
    plt.fill(x_fill4, y_fill4, color='#FDFD6D', alpha=1, zorder=1)  
    plt.fill(x_fill3, y_fill3, color='#FFCB6A', alpha=1, zorder=1)
    plt.fill(x_fill2, y_fill2, color='#FF6A6A', alpha=1, zorder=1) 
    plt.fill(x_fill, y_fill, color='#D482A8', alpha=1, zorder=1)




    if Show_zone_labels == True:
        if plot_size>=300:
            plt.text(500/600*plot_size, 550/600*plot_size, 'A', fontsize=12, color='black')
            plt.text(380/600*plot_size, 550/600*plot_size, 'B', fontsize=12, color='black')
            plt.text(260/600*plot_size, 550/600*plot_size, 'C', fontsize=12, color='black')
            plt.text(180/600*plot_size, 550/600*plot_size, 'D', fontsize=12, color='black')
            plt.text(80/600*plot_size, 550/600*plot_size, 'E', fontsize=12, color='black')

            plt.text(550/600*plot_size, 350/600*plot_size, 'B', fontsize=12, color='black')
            plt.text(550/600*plot_size, 220/600*plot_size, 'C', fontsize=12, color='black')
            plt.text(550/600*plot_size, 140/600*plot_size, 'D', fontsize=12, color='black')
            plt.text(550/600*plot_size, 50/600*plot_size,  'E', fontsize=12, color='black')

    def x_secondary(x):
        return np.round(x / 18)
    def x_secondary_inv(x):
        return np.round(x * 18)
    def y_secondary(y):
        return np.round(y / 18)
    def y_secondary_inv(y):
        return np.round(y * 18)
    

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Define your 5 colors (replace with your desired colors)
    risk_colors = ['#6ABC6A', '#FDFD6D', '#FFCB6A', '#FF6A6A', '#D482A8']  # Example: green, yellow, orange, red, purple
    risk_labels = ["A: None", "B: Mild", "C: Moderate", "D: High", "E: Extreme"]

    cmap = ListedColormap(risk_colors)
    bounds = np.linspace(0, 5, 6)  # 5 sections
    norm = BoundaryNorm(bounds, cmap.N)

    # Create a ScalarMappable and colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(0.5, 5), fraction=0.025, pad=0.13, aspect=40, location='right')
    cbar.ax.set_yticklabels(risk_labels)
    cbar.set_label('Risk Grade',labelpad=5, rotation=90)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(left=False, right=False, length=0,which='both')

    for b in bounds[1:-1]:  # skip first and last bound
        cbar.ax.axhline(b, color='black', linewidth=1)



    if highlight_zone is not None:

        from matplotlib.patches import Polygon as MplPolygon
        polygons = {
                'A': [Path(np.column_stack([x_fill5, y_fill5]))],
                'B': [Path(np.column_stack([x_fill6, y_fill6])),Path(np.column_stack([x_fill4, y_fill4]))],
                'C': [Path(np.column_stack([x_fill7, y_fill7])), Path(np.column_stack([x_fill3, y_fill3]))],
                'D': [Path(np.column_stack([x_fill8, y_fill8])), Path(np.column_stack([x_fill2, y_fill2]))],
                'E': [Path(np.column_stack([x_fill9, y_fill9])), Path(np.column_stack([x_fill, y_fill]))],
            }

        chosen_zone = highlight_zone  # Change to 'A', 'B', 'C', 'D', or 'E' as needed
        zone_colors = {'A': 'red', 'B': 'red', 'C': 'red', 'D': 'red', 'E': 'red'}

        for poly in polygons[chosen_zone]:
            verts = poly.vertices
            patch = MplPolygon(verts, closed=True, fill=False, edgecolor=zone_colors[chosen_zone], linewidth=3, linestyle='-', zorder=30, label=f'Zone {chosen_zone} perimeter')
            ax.add_patch(patch)

    # chosen_zone = 'A'  # Change to 'A', 'B', 'C', 'D', or 'E' as needed
    # zone_colors = {'A': 'green', 'B': 'red', 'C': 'orange', 'D': 'red', 'E': 'purple'}

    # for poly in polygons[chosen_zone]:
    #     verts = poly.vertices
    #     patch = MplPolygon(verts, closed=True, fill=False, edgecolor=zone_colors[chosen_zone], linewidth=1, linestyle='--', zorder=30, label=f'Zone {chosen_zone} perimeter')
    #     ax.add_patch(patch)


    if input_data is not None:
        input_data_x=input_data['REFERENCE']
        input_data_y=input_data['MONITOR']
        plt.scatter(input_data_x, input_data_y, color='white', s=point_size, edgecolor='black', zorder=10)
        epsilon = 1e-8
        MARD = 1/len(input_data_x) * np.sum(np.abs((np.array(input_data_y) - np.array(input_data_x)) / (np.array(input_data_x)))) * 100
        MARD = np.round(MARD, 1)
        # print(MARD)
        # print(len(input_data_x))

        # Calculate absolute relative difference (%)
        rel_diff = np.abs((np.array(input_data_y) - np.array(input_data_x)) / (np.array(input_data_x) + epsilon)) * 100

        # Define bins: 0-5%, 5-10%, ..., 40%+
        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, np.inf]
        bin_labels = [
            "<5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%", "35-40%", ">40%"
        ]

        # Histogram of counts in each bin
        counts, _ = np.histogram(rel_diff, bins=bins)
        percentages = counts / len(rel_diff) * 100
        cumulative_percentages = np.cumsum(percentages)

        # print("Relative difference bins and cumulative percentages:")
        # for i, label in enumerate(bin_labels):
        #     print(f"{label}: {percentages[i]:.1f}% (cumulative: {cumulative_percentages[i]:.1f}%)")




        from shapely.geometry import Polygon

        coords = np.column_stack([x_fill5, y_fill5])
        poly_a = Polygon(coords)

        # Shrink polygon by 1e-4 units inward
        poly_a_smaller = poly_a.buffer(-0.25)

        # Extract new coordinates (ensure they are ordered correctly)
        x_fill5_new, y_fill5_new = poly_a_smaller.exterior.xy

        # Update x_fill5 and y_fill5 with the smaller polygon
        x_fill5 = np.array(x_fill5_new)
        y_fill5 = np.array(y_fill5_new)



        coords = np.column_stack([x_fill3, y_fill3])
        poly_a = Polygon(coords)

        # Shrink polygon by 1e-4 units inward
        poly_a_smaller = poly_a.buffer(-0.25)

        # Extract new coordinates (ensure they are ordered correctly)
        x_fill3_new, y_fill3_new = poly_a_smaller.exterior.xy

        # Update x_fill5 and y_fill5 with the smaller polygon
        x_fill3 = np.array(x_fill3_new)
        y_fill3 = np.array(y_fill3_new)
        



        coords = np.column_stack([x_fill7, y_fill7])
        poly_a = Polygon(coords)

        # Shrink polygon by 1e-4 units inward
        poly_a_smaller = poly_a.buffer(-0.25)

        # Extract new coordinates (ensure they are ordered correctly)
        x_fill7_new, y_fill7_new = poly_a_smaller.exterior.xy

        # Update x_fill5 and y_fill5 with the smaller polygon
        x_fill7 = np.array(x_fill7_new)
        y_fill7 = np.array(y_fill7_new)


        coords = np.column_stack([x_fill, y_fill])
        poly_a = Polygon(coords)

        # Shrink polygon by 1e-4 units inward
        poly_a_smaller = poly_a.buffer(-0.25)

        # Extract new coordinates (ensure they are ordered correctly)
        x_fill_new, y_fill_new = poly_a_smaller.exterior.xy

        # Update x_fill5 and y_fill5 with the smaller polygon
        x_fill = np.array(x_fill_new)
        y_fill = np.array(y_fill_new)


        coords = np.column_stack([x_fill9, y_fill9])
        poly_a = Polygon(coords)

        # Shrink polygon by 1e-4 units inward
        poly_a_smaller = poly_a.buffer(-0.25)

        # Extract new coordinates (ensure they are ordered correctly)
        x_fill9_new, y_fill9_new = poly_a_smaller.exterior.xy

        # Update x_fill5 and y_fill5 with the smaller polygon
        x_fill9 = np.array(x_fill9_new)
        y_fill9 = np.array(y_fill9_new)


        # Define polygons for each zone (combine for B)
        polygons = {
            'E': [Path(np.column_stack([x_fill9, y_fill9])), Path(np.column_stack([x_fill, y_fill]))],
            'D': [Path(np.column_stack([x_fill8, y_fill8])), Path(np.column_stack([x_fill2, y_fill2]))],
            'C': [Path(np.column_stack([x_fill7, y_fill7])), Path(np.column_stack([x_fill3, y_fill3]))],
            'B': [Path(np.column_stack([x_fill6, y_fill6])),Path(np.column_stack([x_fill4, y_fill4]))],
            'A': [Path(np.column_stack([x_fill5, y_fill5]))],    
        }

        zone_counts = {zone: 0 for zone in polygons}
        points = np.column_stack([input_data_x, input_data_y])
        assigned = np.zeros(len(points), dtype=bool)  # Track if a point has been assigned

        # Assign each point to the first zone it falls into (A, B, C, D, E order)
        for zone in ['E', 'D', 'C', 'B', 'A']:
            inside = np.zeros(len(points), dtype=bool)
            for poly in polygons[zone]:
                inside |= poly.contains_points(points)
            # Only count points not already assigned
            new_assignments = inside & (~assigned)
            zone_counts[zone] = np.sum(new_assignments)
            assigned |= new_assignments  # Mark these points as assigned

        zone_weights = {
        'A': 1.0,  # Default weight
        'B': 1.2,   # 50% higher influence than A
        'C': 1.4, 
        'D': 1.6,
        'E': 1.8
        }


        from shapely.geometry import Point, MultiPolygon, Polygon

        # Convert each zone's Paths to Shapely MultiPolygons
        zone_multipolygons = {}
        for zone in polygons:
            zone_polys = []
            for path in polygons[zone]:
                # Extract vertices from the Path object (assuming path.vertices exists)
                vertices = path.vertices
                coords = [(x, y) for x, y in vertices]
                zone_polys.append(Polygon(coords))
            zone_multipolygons[zone] = MultiPolygon(zone_polys)

        unassigned_indices = np.where(~assigned)[0]
        for idx in unassigned_indices:
            x, y = points[idx]
            point = Point(x, y)
            closest_zone = None
            min_weighted_dist = float('inf')
            
            # Check weighted distance to each zone
            for zone in ['A', 'B', 'C', 'D', 'E']:
                dist = zone_multipolygons[zone].distance(point)
                weighted_dist = dist / zone_weights[zone]  # Key change: divide by weight
                if weighted_dist < min_weighted_dist:
                    min_weighted_dist = weighted_dist
                    closest_zone = zone
            
            if closest_zone is not None:
                zone_counts[closest_zone] += 1
                assigned[idx] = True

        
        total_points = len(points)
        if save_graph==True:
            plt.xlim(0, plot_size)
            plt.ylim(0, plot_size)
            plt.xlabel('Reference Glucose (mg/dL)')
            plt.ylabel('Monitor Glucose (mg/dL)')
            if graph_title is not None:
                plt.title(graph_title, fontsize=14)

            secax_x = ax.secondary_xaxis('top', functions=(x_secondary, x_secondary_inv))
            secax_x.set_xlabel('Reference Glucose (mmol/L)')
            secax_y = ax.secondary_yaxis('right', functions=(y_secondary, y_secondary_inv))
            secax_y.set_ylabel('Monitor Glucose (mmol/L)')
            if graph_title is not None:
                plt.title(graph_title, fontsize=14)
                downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
                save_name=graph_title+".pdf"
                pdf_path = os.path.join(downloads_folder, save_name)
                pdf = PdfPages(pdf_path)
                pdf.savefig(fig)  # Save the plot page
                plt.close(fig)
            
            else:
                downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
                
                pdf_path = os.path.join(downloads_folder, "output.pdf")
                pdf = PdfPages(pdf_path)
                pdf.savefig(fig)  # Save the plot page
                plt.close(fig)
            

            # --- Relative difference bins table ---
            rel_diff_table = pd.DataFrame({
                "Bin": bin_labels,
                "Percentage (%)": np.round(percentages, 1),
                "Cumulative (%)": np.round(cumulative_percentages, 1)
            })
            # print("\nRelative difference bins and cumulative percentages:")
            # print(rel_diff_table.to_string(index=False))

            # --- Zone counts table ---
            zone_table = pd.DataFrame({
                "Zone": ['A', 'B', 'C', 'D', 'E'],
                "Count": [zone_counts[z] for z in ['A', 'B', 'C', 'D', 'E']],
                "Percentage (%)": [np.round(zone_counts[z] / total_points * 100, 1) for z in ['A', 'B', 'C', 'D', 'E']]
            })
            # print("\nPoints in each zone:")
            # print(zone_table.to_string(index=False))
            # Save zone_table, rel_diff_table, and MARD on the same page
            fig2, axarr = plt.subplots(3, 1, figsize=(8.5, 11), gridspec_kw={'height_ratios': [1,5, 5]})
            mard_str = f"{MARD:.1f}"
            pza = zone_counts['A'] / total_points * 100
            ab = (zone_counts['A'] + zone_counts['B']) / total_points * 100

            summary_data = [
                ["MARD (%)", mard_str],
                ["Clinically Accurate (PzA) (%)", f"{pza:.1f}"],
                ["Clinically Acceptable (A+B) (%)", f"{ab:.1f}"]
            ]

            axarr[0].axis('off')
            summary_table = axarr[0].table(
                cellText=summary_data,
                colLabels=None,
                cellLoc='center',
                loc='center'
            )
            summary_table.auto_set_font_size(False)
            summary_table.set_fontsize(12)
            summary_table.scale(1.2, 1.2)
            # Zone table
            axarr[1].axis('off')
            axarr[1].text(0.5, 0.7, "Risk Zone Comparison", fontsize=13, ha='center', va='bottom', transform=axarr[1].transAxes)
            table2 = axarr[1].table(cellText=zone_table.values,
                                    colLabels=zone_table.columns,
                                    loc='center',
                                    cellLoc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(12)
            table2.scale(1.2, 1.2)
            # Relative difference table
            axarr[2].axis('off')
            axarr[2].text(0.5, 0.82, "Relative Error Ranges", fontsize=13, ha='center', va='bottom', transform=axarr[2].transAxes)
            table3 = axarr[2].table(cellText=rel_diff_table.values,
                                    colLabels=rel_diff_table.columns,
                                    loc='center',
                                    cellLoc='center')
            table3.auto_set_font_size(False)
            table3.set_fontsize(12)
            table3.scale(1.2, 1.2)
            plt.tight_layout(rect=[0,0,1,0.95])  # Reduce top/bottom margin
            pdf.savefig(fig2)
            plt.close(fig2)

            pdf.close()
        
        else:
            
        
            # for zone in ['A', 'B', 'C', 'D', 'E']:
            #     percent = zone_counts[zone] / total_points * 100
            #     print(f"Number of points in zone {zone}: {zone_counts[zone]} ({percent:.1f}%)")

            # unassigned_indices = np.where(~assigned)[0]
            # print("Indices of points not in any zone:", unassigned_indices)

            # --- MARD ---
            print(f"MARD: {MARD:.1f}%")

            # --- Relative difference bins table ---
            rel_diff_table = pd.DataFrame({
                "Bin": bin_labels,
                "Percentage (%)": np.round(percentages, 1),
                "Cumulative (%)": np.round(cumulative_percentages, 1)
            })
            print("\nRelative difference bins and cumulative percentages:")
            print(rel_diff_table.to_string(index=False))

            # --- Zone counts table ---
            zone_table = pd.DataFrame({
                "Zone": ['A', 'B', 'C', 'D', 'E'],
                "Count": [zone_counts[z] for z in ['A', 'B', 'C', 'D', 'E']],
                "Percentage (%)": [np.round(zone_counts[z] / total_points * 100, 1) for z in ['A', 'B', 'C', 'D', 'E']]
            })
            print("\nPoints in each zone:")
            print(zone_table.to_string(index=False))
            plt.xlim(0, plot_size)
            plt.ylim(0, plot_size)
            if graph_title is not None:
                plt.title(graph_title, fontsize=14)
            plt.xlabel('Reference Glucose (mg/dL)')
            plt.ylabel('Monitor Glucose (mg/dL)')
            secax_x = ax.secondary_xaxis('top', functions=(x_secondary, x_secondary_inv))
            secax_x.set_xlabel('Reference Glucose (mmol/L)')
            secax_y = ax.secondary_yaxis('right', functions=(y_secondary, y_secondary_inv))
            secax_y.set_ylabel('Monitor Glucose (mmol/L)')
            plt.show()


    else:
        plt.xlim(0, plot_size)
        plt.ylim(0, plot_size)
        plt.xlabel('Reference Glucose (mg/dL)')
        plt.ylabel('Monitor Glucose (mg/dL)')
        secax_x = ax.secondary_xaxis('top', functions=(x_secondary, x_secondary_inv))
        secax_x.set_xlabel('Reference Glucose (mmol/L)')
        secax_y = ax.secondary_yaxis('right', functions=(y_secondary, y_secondary_inv))
        secax_y.set_ylabel('Monitor Glucose (mmol/L)')
        plt.show()

if __name__ =="__main__":
    plot_DTSEG(input_data=None, plot_size=600, point_size=5, save_graph=True, graph_title=None)
