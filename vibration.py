from gpiozero import OutputDevice
from time import sleep

# Indichiamo che il segnale (IN) è collegato al GPIO 17
motore = OutputDevice(17)

print("Inizio test vibrazione...")

try:
    # Ripete il ciclo 3 volte
    for i in range(3):
        print("Motore ACCESO")
        motore.on()   # Invia il segnale HIGH al pin IN
        sleep(1)      # Attende 1 secondo
        
        print("Motore SPENTO")
        motore.off()  # Invia il segnale LOW al pin IN
        sleep(1)      # Attende 1 secondo

    print("Test completato.")

except KeyboardInterrupt:
    # Se premi Ctrl+C, spegne il motore e chiude il programma in sicurezza
    motore.off()
    print("\nProgramma interrotto dall'utente.")
