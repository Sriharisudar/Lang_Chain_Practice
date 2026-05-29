import os
from PyPDF2 import PdfMerger  # Or PdfFileMerger in older versions

# Path to the folder containing PDF files
folder_path = r"D:\Movies\New folder"

# List all PDF files in the folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
pdf_files.sort()  # Optional: Sort files alphabetically

# Create a PdfMerger object
merger = PdfMerger()

# Append each PDF file
for pdf in pdf_files:
    full_path = os.path.join(folder_path, pdf)
    merger.append(full_path)

# Output file
output_file = os.path.join(folder_path, "combined_output.pdf")
merger.write(output_file)
merger.close()

print(f"Combined PDF saved as: {output_file}")
