import random

def hangman():
    word_list = ["blackforest", "truffle", "vanilla", "strawberry", "deathbychocolate", "blueberry", "butterscotch"]
    word = random.choice(word_list).lower()
    guessed_letter = []
    incorrect_guesses = 0
    max_guesses = 6

    print("Hangman Starts!....1...2...3...")
    print("_ " * len(word))
    
    while True:
        guess = input("Guess a letter: ").lower()
        
        if len(guess) != 1 or not guess.isalpha():
            print("Invalid input.")
            continue
        
        if guess in guessed_letter:
            print("You've already guessed that letter.")
            continue
        
        guessed_letter.append(guess)

        if guess in word:
            print("Correct guess!")
        else:
            print("Incorrect guess!")
            incorrect_guesses += 1
        
        print("Guessed letters:", " ".join(guessed_letter))
        
        displayed_word = ""
        for letter in word:
            if letter in guessed_letter:
                displayed_word += letter + " "
            else:
                displayed_word += "_ "
                
        print(displayed_word)
        
        if all(letter in guessed_letter for letter in word):
            print(" You Won !")
            break
        
        if incorrect_guesses == max_guesses:
            print("Game over!")
            print("The word was:", word)
            break
        
hangman()