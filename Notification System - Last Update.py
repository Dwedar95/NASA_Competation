import requests
import geocoder
import smtplib
from email.mime.text import MIMEText
import datetime
import time
import threading

# API key for N2YO
api_key = "29GXSU-H9RJRB-VYQCJP-5CK2"  # Insert your N2YO API key here

# NORAD IDs for Landsat 8 and Landsat 9
landsat8_norad_id = 39084
landsat9_norad_id = 49260

# Function to get the satellite currently above the observer's position
def get_satellite_above(observer_lat, observer_lng, observer_alt):
    url = f"https://api.n2yo.com/rest/v1/satellite/above/{observer_lat}/{observer_lng}/{observer_alt}/90/18&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['above']:
            return data['above'][0]  # Return the first satellite found
        else:
            return None
    else:
        print(f"Error in API request: {response.status_code}")
        return None

# Function to convert Unix timestamp to UTC datetime
def unix_to_utc(unix_time):
    return datetime.datetime.utcfromtimestamp(unix_time)

# Function to get the next pass of Landsat 8 and Landsat 9
def get_next_landsat_passes(observer_lat, observer_lng, observer_alt):
    results = {}
    
    # N2YO API for Landsat 8
    url_landsat8 = f"https://api.n2yo.com/rest/v1/satellite/visualpasses/{landsat8_norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/90/1/&apiKey={api_key}"
    response_landsat8 = requests.get(url_landsat8)
    if response_landsat8.status_code == 200:
        data_landsat8 = response_landsat8.json()
        if data_landsat8['passes']:
            results['Landsat 8'] = data_landsat8['passes'][0]  # Return the first pass for Landsat 8
        else:
            results['Landsat 8'] = None
    else:
        print(f"Error in API request for Landsat 8: {response_landsat8.status_code}")

    # N2YO API for Landsat 9
    url_landsat9 = f"https://api.n2yo.com/rest/v1/satellite/visualpasses/{landsat9_norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/90/1/&apiKey={api_key}"
    response_landsat9 = requests.get(url_landsat9)
    if response_landsat9.status_code == 200:
        data_landsat9 = response_landsat9.json()
        if data_landsat9['passes']:
            results['Landsat 9'] = data_landsat9['passes'][0]  # Return the first pass for Landsat 9
        else:
            results['Landsat 9'] = None
    else:
        print(f"Error in API request for Landsat 9: {response_landsat9.status_code}")

    return results

# Function to get the current position (latitude, longitude, altitude)
def get_current_position():
    g = geocoder.ip('me')  # Get the position based on IP address
    observer_lat = g.latlng[0]
    observer_lng = g.latlng[1]
    observer_alt = 0  # Altitude is optional, set to 0 here, can be adjusted
    return observer_lat, observer_lng, observer_alt

# Function to send email notifications
def send_email(subject, body, to_email):
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "mohamedhusien09@gmail.com"  # Replace with your Gmail address
    password = "wktx cbmf bpjr tihl"  # Replace with your Gmail App Password

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        # Create a secure connection with the server and send email
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)  # Log in to your Gmail account
        server.sendmail(sender_email, to_email, msg.as_string())  # Send the email
        server.quit()  # Close the connection
        print(f"Notification sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to schedule a reminder email
def schedule_reminder(pass_time, subject, email_body, recipient_email):
    # Calculate how long to wait before sending the reminder email
    one_hour_before = pass_time - datetime.timedelta(hours=1)
    time_to_wait = (one_hour_before - datetime.datetime.utcnow()).total_seconds()

    if time_to_wait > 0:
        time.sleep(time_to_wait)  # Wait until one hour before the pass
        send_email(subject, email_body, recipient_email)  # Send reminder email

# Main program
if __name__ == "__main__":
    # Input specific location or use current position
    use_current_position = input("Do you want to use your current position? (yes/no): ").strip().lower()
    
    if use_current_position == 'yes':
        # Get the user's current position
        observer_lat, observer_lng, observer_alt = get_current_position()
        print(f"Your current position: Lat: {observer_lat}, Lng: {observer_lng}, Alt: {observer_alt}")
    else:
        # Input desired location from user
        observer_lat = float(input("Enter latitude: "))
        observer_lng = float(input("Enter longitude: "))
        observer_alt = float(input("Enter altitude (in meters): "))

    # Get the next pass for Landsat 8 and Landsat 9
    next_passes = get_next_landsat_passes(observer_lat, observer_lng, observer_alt)
    email_body = ""

    # Process Landsat 8 Pass
    if next_passes['Landsat 8']:
        pass_time_utc = unix_to_utc(next_passes['Landsat 8']['startUTC'])
        print(f"Next pass of Landsat 8: {pass_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        email_body += f"Hello,\n\nYou requested tracking of Landsat 8. It is coming soon, be excited and ready!\nNext pass: {pass_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\nRegards,\n"
        
        # Send immediate email notification
        subject = "Next Landsat 8 Pass Notification"
        recipient_email = input("Enter your email address to receive notifications: ").strip()
        send_email(subject, email_body, recipient_email)
        
        # Schedule reminder email one hour before the pass
        reminder_subject = "Landsat 8 Pass Reminder"
        reminder_email_body = email_body
        threading.Thread(target=schedule_reminder, args=(pass_time_utc, reminder_subject, reminder_email_body, recipient_email)).start()

    else:
        print("No pass found for Landsat 8.")

    # Process Landsat 9 Pass
    if next_passes['Landsat 9']:
        pass_time_utc = unix_to_utc(next_passes['Landsat 9']['startUTC'])
        print(f"Next pass of Landsat 9: {pass_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        email_body = f"Hello,\n\nYou requested tracking of Landsat 9. It is coming soon, be excited and ready!\nNext pass: {pass_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\nRegards,\n"
        
        # Send immediate email notification
        subject = "Next Landsat 9 Pass Notification"
        send_email(subject, email_body, recipient_email)
        
        # Schedule reminder email one hour before the pass
        reminder_subject = "Landsat 9 Pass Reminder"
        reminder_email_body = email_body
        threading.Thread(target=schedule_reminder, args=(pass_time_utc, reminder_subject, reminder_email_body, recipient_email)).start()

    else:
        print("No pass found for Landsat 9.")


