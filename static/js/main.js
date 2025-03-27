// Common utility functions
const utils = {
    // Format date
    formatDate: function(date) {
        return new Date(date).toLocaleString();
    },
    
    // Show loading spinner
    showLoading: function(element) {
        element.html('<div class="spinner"></div>');
    },
    
    // Hide loading spinner
    hideLoading: function(element) {
        element.empty();
    },
    
    // Show alert message
    showAlert: function(message, type = 'info') {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        $('#alertContainer').html(alertHtml);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            $('.alert').alert('close');
        }, 5000);
    },
    
    // Update map marker
    updateMapMarker: function(map, marker, lat, lng) {
        map.setView([lat, lng], 12);
        marker.setLatLng([lat, lng]);
    },
    
    // Get weather data
    getWeatherData: function(lat, lng) {
        return $.ajax({
            url: `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=YOUR_API_KEY&units=imperial`,
            method: 'GET'
        });
    }
};

// Initialize tooltips
$(function () {
    $('[data-bs-toggle="tooltip"]').tooltip();
});

// Handle form validation
function validateForm(formId) {
    const form = $(`#${formId}`);
    let isValid = true;
    
    form.find('input[required], select[required]').each(function() {
        if (!$(this).val()) {
            isValid = false;
            $(this).addClass('is-invalid');
        } else {
            $(this).removeClass('is-invalid');
        }
    });
    
    return isValid;
}

// Handle map initialization
function initMap(elementId, lat, lng, zoom = 12) {
    const map = L.map(elementId).setView([lat, lng], zoom);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    return map;
}

// Handle real-time updates
function startRealTimeUpdates() {
    // Update emergency vehicle positions every 30 seconds
    setInterval(() => {
        $.get('/api/emergency-vehicles', function(data) {
            updateEmergencyVehicles(data);
        });
    }, 30000);
    
    // Update weather data every 5 minutes
    setInterval(() => {
        if (currentLocation) {
            utils.getWeatherData(currentLocation.lat, currentLocation.lng)
                .done(function(data) {
                    updateWeatherData(data);
                });
        }
    }, 300000);
}

// Handle emergency alerts
function handleEmergencyAlert(alert) {
    // Show notification
    if (Notification.permission === "granted") {
        new Notification("Emergency Alert", {
            body: alert.message,
            icon: "/static/img/alert-icon.png"
        });
    }
    
    // Update alerts list
    const alertHtml = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle"></i> ${alert.message}
        </div>
    `;
    $('.alert-list').prepend(alertHtml);
}

// Initialize the application
$(document).ready(function() {
    // Add alert container if it doesn't exist
    if (!$('#alertContainer').length) {
        $('body').prepend('<div id="alertContainer" class="position-fixed top-0 end-0 p-3" style="z-index: 1050"></div>');
    }
    
    // Start real-time updates
    startRealTimeUpdates();
    
    // Handle form submissions
    $('form').on('submit', function(e) {
        if (!validateForm($(this).attr('id'))) {
            e.preventDefault();
            utils.showAlert('Please fill in all required fields', 'danger');
        }
    });
}); 