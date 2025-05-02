import sys
from PyQt5.QtWidgets import QApplication
from Noise import Noiser


if __name__ == "__main__":
    app=QApplication(sys.argv)
    window=Noiser()
    window.show()    
    sys.exit(app.exec_())