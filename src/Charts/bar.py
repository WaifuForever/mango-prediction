import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


class Bar:
    def __init__(self, model_name, scores):
        self.model_name = model_name
        self.scores = scores
      

    # GLOBAL CONSTANTS


    def plot_model_results(self, model_name, scores):
        fig, ax1 = plt.subplots(figsize=(9, 7))  # Create the figure
        fig.subplots_adjust(left=0.115, right=0.88)
        fig.canvas.manager.set_window_title('Accuracy Chart')
        test_names = ["Rotten", "Good", "Total"]
        pos = np.arange(len(test_names))

        rects = ax1.barh(pos, scores,
                        align='center',
                        height=0.5,
                        tick_label=test_names)

        ax1.set_title(model_name)

        ax1.set_xlim([0, 100])
        ax1.xaxis.set_major_locator(MaxNLocator(11))
        ax1.xaxis.grid(True, linestyle='--', which='major',
                    color='grey', alpha=.25)

        # Plot a solid vertical gridline to highlight the median position
        ax1.axvline(50, color='grey', alpha=0.25)

        # Set the right-hand Y-axis ticks and labels
        ax2 = ax1.twinx()

        # Set the tick locations
        ax2.set_yticks(pos)
        # Set equal limits on both yaxis so that the ticks line up
        ax2.set_ylim(ax1.get_ylim())

        # Set the tick labels
        

        ax2.set_ylabel('Hits Score')

        xlabel = ('Percentile of Correct Answers of the Model {model_name}\n'
                'Cohort Size: {cohort_size}')
        ax1.set_xlabel(xlabel.format(model_name=model_name, cohort_size=scores[2]))

        rect_labels = []
        # Lastly, write in the ranking inside each bar to aid in interpretation
        for rect in rects:
            # Rectangle widths are already integer-valued but are floating
            # type, so it helps to remove the trailing decimal point and 0 by
            # converting width to int type
            width = int(rect.get_width())

            # The bars aren't wide enough to print the ranking inside
            if width < 40:
                # Shift the text to the right side of the right edge
                xloc = 5
                # Black against white background
                clr = 'black'
                align = 'left'
            else:
                # Shift the text to the left side of the right edge
                xloc = -5
                # White on magenta
                clr = 'white'
                align = 'right'

            # Center the text vertically in the bar
            yloc = rect.get_y() + rect.get_height() / 2
            label = ax1.annotate(
                text='',
                xy=(width, yloc), xytext=(xloc, 0),
                textcoords="offset points",
                horizontalalignment=align, verticalalignment='center',
                color=clr, weight='bold', clip_on=True)
            rect_labels.append(label)

        # Make the interactive mouse over give the bar title
      
        # Return all of the artists created
        return {'fig': fig,
                'ax': ax1,
                'ax_right': ax2,
                'bars': rects,
                'perc_labels': rect_labels}

    def run(self):
        
        cohort_size = 62  # The number of other 2nd grade boys

        arts = self.plot_model_results(self.model_name, self.scores)
        MODEL_DIR = "./Models/"
        plt.savefig(MODEL_DIR + self.model_name + '/'+ self.model_name + '_result' +'.png')    
        plt.show()