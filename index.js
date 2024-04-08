$(document).ready(function() {
    // Display the selected file name immediately when a file is chosen
    $('.custom-file-input').on('change', function() {
        let fileName = $(this).val().split('\\').pop();
        $(this).siblings('.custom-file-label').html(fileName).addClass("selected");
    });

    $('#uploadButton').on('click', function(event) {
        event.preventDefault(); // Prevent the form from submitting traditionally
        processFile();
    });

    function processFile() {
        const fileInput = $('#customFile')[0];
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            // Check if the file is an Excel file
            if (file.type === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" || 
                file.type === "application/vnd.ms-excel") {
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    // Here, you would process the Excel file. Since parsing Excel files
                    // in the browser requires a library like SheetJS (xlsx), this part is
                    // conceptual and assumes such functionality is available.
                    // Example: const data = parseExcel(e.target.result);
                    
                    // This is where you'd check for required columns and handle the data.
                    // For demonstration, we're directly proceeding to set LocalStorage and redirect.
                    localStorage.setItem("uploadStatus", "File processed successfully.");
                    // localStorage.setItem("initialData", JSON.stringify(data.slice(0, 5))); // Example

                    window.location.href = 'result.html';
                };

                reader.readAsBinaryString(file); // Or readAsArrayBuffer; depends on how you parse Excel files
            } else {
                alert("Please upload an Excel file.");
            }
        } else {
            alert("No file selected. Please select a file.");
        }
    }
});
