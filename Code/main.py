import sys
from PyQt5.QtWidgets import QApplication
from threshold import Threshold


if __name__ == "__main__":
    app=QApplication(sys.argv)
    window=Threshold()
    window.show()    
    sys.exit(app.exec_())