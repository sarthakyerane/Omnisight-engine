import pystray
from PIL import Image, ImageDraw
import threading

class TrayApp:
    def __init__(self):
        self.is_paused = False
        self.icon = None
        self.stop_event = threading.Event()
        
    def create_image(self, color):
        """Creates a simple colored circle image for the tray icon."""
        image = Image.new('RGB', (64, 64), color=(255, 255, 255))
        dc = ImageDraw.Draw(image)
        dc.ellipse((16, 16, 48, 48), fill=color)
        return image

    def on_pause_resume(self, icon, item):
        self.is_paused = not self.is_paused
        # Update icon color: green for active, red for paused
        color = 'red' if self.is_paused else 'green'
        icon.icon = self.create_image(color)

    def on_quit(self, icon, item):
        self.stop_event.set()
        icon.stop()

    def run(self):
        menu = pystray.Menu(
            pystray.MenuItem(lambda text: 'Resume' if self.is_paused else 'Pause', self.on_pause_resume),
            pystray.MenuItem('Quit', self.on_quit)
        )
        self.icon = pystray.Icon("AmbientScreen", self.create_image('green'), "Ambient Screen Memory", menu)
        
        # This will block the thread it runs in
        self.icon.run()
