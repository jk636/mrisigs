import matplotlib.pyplot as plt
import matplotlib.style as style

def lplot(xlab=None, ylab=None, ptitle=None, ax_limits=None, gridon=True, legend_on=False):
    """
    Applies common plot properties.
    """
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if ptitle is not None:
        plt.title(ptitle)
    if ax_limits is not None:
        # Ensure ax_limits is a list/tuple of 4 elements: [xmin, xmax, ymin, ymax]
        if isinstance(ax_limits, (list, tuple)) and len(ax_limits) == 4:
            plt.axis(ax_limits)
        else:
            print("Warning: ax_limits should be a list or tuple of 4 elements [xmin, xmax, ymin, ymax].")
    if gridon: # gridon is already boolean
        plt.grid(gridon) 
    if legend_on:
        plt.legend()

def setprops():
    """
    Sets some default Matplotlib styles and properties.
    """
    # Try a few common styles, fall back if none are available
    try:
        style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            style.use('ggplot')
        except:
            print("Matplotlib styles 'seaborn-v0_8-whitegrid' and 'ggplot' not found. Using default.")
    
    # Add any other common properties if needed
    plt.rcParams['figure.dpi'] = 100 # Example property
    # Add more properties as desired, e.g.:
    # plt.rcParams['lines.linewidth'] = 2
    # plt.rcParams['axes.labelsize'] = 'large'
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Example fallback font
