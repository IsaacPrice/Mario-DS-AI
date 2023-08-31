// This function fetches the game data and updates the webpage
async function fetchDataAndUpdatePage() {
    try {
      const response = await fetch('/get_window_data');  // Replace this URL with your Flask API endpoint if it's different
      const data = await response.json();  // Convert the response data to a JSON object
  
      // Update the webpage with the new data
      document.getElementById('velocity').textContent = `Mario's velocity: [${data.game_data.velocity}]`;
      document.getElementById('reward').textContent = `Reward: ${data.game_data.reward}`;
    } catch (error) {
      console.error('An error occurred:', error);
    }
  }
  
  // Fetch the data every second
  setInterval(fetchDataAndUpdatePage, 1000);
  