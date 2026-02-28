from gpiozero import OutputDevice, Button
from signal import pause

# Configuriamo i pin
motore = OutputDevice(17)
pulsante = Button(27)

print("Premi il pulsante per far vibrare il motore!")
print("Rilascialo per fermarlo. Premi Ctrl+C per chiudere il programma.")

# Colleghiamo gli eventi del pulsante direttamente ai comandi del motore
pulsante.when_pressed = motore.on
pulsante.when_released = motore.off

try:
    # La funzione pause() mette il programma in "attesa infinita"
    # senza consumare la CPU del Raspberry Pi.
    # Il codice reagirà solo quando premi il pulsante.
    pause()
except KeyboardInterrupt:
    print("\nProgramma chiuso. Arrivederci!")
