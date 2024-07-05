

l
def load_documents(filenames):
   """Loads documents from a list of filenames.

  Args:
      filenames: A list of strings representing file paths.

  Returns:
      A string containing the combined text of all documents.
  """
documents = ""
for filename in filenames:
    with open(filename, 'r') as f:
      documents += f.read() + "\n"  # Add newline for better separation
    return documents

def answer_question(documents, question):
    """Answers a question based on the provided documents.

    This is a simple implementation that searches for keywords in the question
    within the documents.

    Args:
    documents: A string containing the combined text of all documents.
    question: A string representing the user's question.

  Returns:
      A string containing the answer or "Sorry, I couldn't find an answer".
  """
  lowercase_question = question.lower()
  if lowercase_question in documents.lower():
    return f"Answer: Found a match in the documents for your question."
  else:
    return "Sorry, I couldn't find an answer to your question in the documents."

# Load documents
filenames = ["document1.txt", "document2.txt"]  # Replace with your filenames
documents = load_documents(filenames)

# User interaction loop
while True:
  question = input("Ask a question about the documents (or 'quit' to exit): ")
  if question.lower() == "quit":
    break
  answer = answer_question(documents, question)
  print(answer)

print("Thanks for using the document Q&A system!")
