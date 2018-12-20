import subprocess
import time

player1 = subprocess.Popen(['vlc', 'music/deck-the-halls.mp3', '--intf', 'dummy'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

player2 = subprocess.Popen(['vlc', 'music/jingle-bells.mp3', '--intf', 'dummy'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

player3 = subprocess.Popen(['vlc', 'music/what-child-is-this.mp3', '--intf', 'dummy'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

players = [player1, player2, player3]

time.sleep(10)
[p.kill() for p in players]
